from __future__ import annotations

import os
from typing import Literal, OrderedDict, Union, override

import lightning as L
import segmentation_models_pytorch as smp
import torch
from dataset.dataset import CineDataset, get_trainval_data_subsets
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
    LRSchedulerCallable,
    OptimizerCallable,
)
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from models.two_plus_one import Unet
from segmentation_models_pytorch.losses.dice import DiceLoss
from torch import nn
from torch.nn import functional as F
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.segmentation import GeneralizedDiceScore
from torchvision.transforms import v2
from torchvision.transforms.transforms import Compose
from warmup_scheduler import GradualWarmupScheduler

BATCH_SIZE_TRAIN = 4

BATCH_SIZE_VAL = 4
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
LEARNING_RATE = 1e-4
NUM_FRAMES = 5
SEED_CUS = 1  # RNG seed.
torch.set_float32_matmul_precision("medium")


class LightningGradualWarmupScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        multiplier: int = 2,
        total_epoch: int = 5,
        T_max=50,
        after_scheduler=None,
    ):
        self.optimizer = optimizer
        after_scheduler = (
            after_scheduler if after_scheduler else CosineAnnealingLR(optimizer, T_max)
        )
        self.scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=multiplier,
            total_epoch=total_epoch,
            after_scheduler=after_scheduler,
        )

    def step(self, epoch=None, metrics=None):
        return self.scheduler.step(epoch, metrics)


class UnetLightning(L.LightningModule):
    def __init__(
        self,
        metric: Metric | None = None,
        loss: nn.Module | None = None,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        num_frames: int = 5,
        weights_from_ckpt_path: str | None = None,
        optimizer: OptimizerCallable = AdamW,
        scheduler: LRSchedulerCallable | str = "gradual_warmup_scheduler",
        multiplier: int = 2,
        total_epochs: int = 50,
        alpha: float = 1.0,
        _beta: float = 0.0,
    ):
        """A LightningModule wrapper for the modified Unet for the two-plus-one
        architecture.

        Args:
            loss: Loss/Criterion callable.
            metric: Metric object for tracking performance.
            encoder_name: Encoder/Backbone model name for the Unet.
            encoder_depth: Number of encoder blocks in the Unet.
            encoder_weights: Weights for the encoder/backbone model.
            in_channels: Number of channels in input.
            classes: Number of classes for output.
            num_frames: Number of frames to use from dataset.
            optimizer: Instantiable optimizer for model parameters.
            scheduler: Instantiable LR Scheduler or string (for custom schedulers)
            multiplier: For the GradualWarmupScheduler LR Scheduler.
            total_epochs: For the CosineAnnealingLR portion of the
            GradualWarmupScheduler.
            alpha: Loss scaler.
            _beta: (Unused) Loss scaler.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["metric", "loss"])
        self.model = Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            num_frames=num_frames,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = (
            loss if loss else DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        )
        self.metric = (
            metric
            if metric
            else GeneralizedDiceScore(
                num_classes=classes,
                per_class=True,
                include_background=True,
                weight_type="linear",
            )
        )

        self.multiplier = multiplier
        self.total_epochs = total_epochs

        self.alpha = alpha

        # Attempts to load checkpoint if provided.
        self.weights_from_ckpt_path = weights_from_ckpt_path
        if self.weights_from_ckpt_path:
            ckpt = torch.load(self.weights_from_ckpt_path)
            try:
                self.load_state_dict(ckpt["state_dict"])
            except KeyError:
                # HACK: So that legacy checkpoints can be loaded.
                try:
                    new_state_dict = OrderedDict()
                    for k, v in ckpt.items():
                        name = k[7:]  # remove 'module.' of dataparallel
                        new_state_dict[name] = v
                    self.model.load_state_dict(new_state_dict)
                except RuntimeError as e:
                    raise e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ) -> torch.Tensor:
        images, masks, _ = batch
        bs = images.shape[0]
        images = images.permute(1, 0, 2, 3, 4)
        images = images.to(DEVICE, dtype=torch.float32)
        masks = masks.to(DEVICE).long()

        with torch.autocast(device_type=self.device.type):
            masks_proba = self.model(images)

        loss_seg = self.alpha * self.loss(masks_proba, masks)
        loss_all = loss_seg
        self.log(f"train_loss", loss_all.item(), batch_size=bs, on_epoch=True)

        if isinstance(self.metric, GeneralizedDiceScore):
            masks = masks.squeeze(1)  # BS x H x W

            masks_one_hot = F.one_hot(masks, num_classes=4).permute(
                0, -1, 1, 2
            )  # BS x C x H x W
            masks_preds = masks_proba.argmax(dim=1)  # BS x H x W

            masks_preds_one_hot = F.one_hot(masks_preds, num_classes=4).permute(
                0, -1, 1, 2
            )  # BS x C x H x W

            class_distribution = masks_one_hot.sum(dim=[0, 2, 3])  # 1 x C
            class_distribution = class_distribution.div(
                class_distribution[1:].sum()
            ).squeeze()

            metric: torch.Tensor = self.metric(masks_preds_one_hot, masks_one_hot)
            weighted_avg = metric[1:] @ class_distribution[1:]

            self.log(f"train_dice_(weighted_avg)", weighted_avg.item(), batch_size=bs)
            self.log(f"train_dice_(macro_avg)", metric.mean().item(), batch_size=bs)

            for i, class_metric in enumerate(metric.detach().cpu()):
                if i == 0:  # NOTE: Skips background class.
                    continue
                self.log(
                    f"train_dice_class_{i}",
                    class_metric.item(),
                    batch_size=bs,
                )
        return loss_all

    def on_train_epoch_end(self):
        self.metric.reset()

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ):
        self._shared_eval(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        self.metric.reset()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, str], batch_idx: int):
        self._shared_eval(batch, batch_idx, "test")

    def on_test_epoch_end(self):
        self.metric.reset()

    def _shared_eval(
        self, batch: tuple[torch.Tensor, torch.Tensor, str], batch_idx: int, prefix: str
    ):
        images, masks, _ = batch
        bs = images.shape[0] if len(images.shape) > 3 else 1
        images = images.permute(1, 0, 2, 3, 4)
        images = images.to(DEVICE, dtype=torch.float32)
        masks = masks.to(DEVICE).long()
        masks_proba = self.model(images)  # BS x C x H x W
        loss_seg = self.alpha * self.loss(masks_proba, masks)
        loss_all = loss_seg
        self.log(f"{prefix}_loss", loss_all.detach().cpu().item(), batch_size=bs)

        if isinstance(self.metric, GeneralizedDiceScore):
            masks = masks.squeeze(1)  # BS x H x W

            masks_one_hot = F.one_hot(masks, num_classes=4).permute(
                0, -1, 1, 2
            )  # BS x C x H x W
            masks_preds = masks_proba.argmax(dim=1)  # BS x H x W

            masks_preds_one_hot = F.one_hot(masks_preds, num_classes=4).permute(
                0, -1, 1, 2
            )  # BS x C x H x W

            class_distribution = masks_one_hot.sum(dim=[0, 2, 3])  # 1 x C
            class_distribution = class_distribution.div(
                class_distribution[1:].sum()
            ).squeeze()

            metric: torch.Tensor = self.metric(masks_preds_one_hot, masks_one_hot)
            weighted_avg = metric[1:] @ class_distribution[1:]

            self.log(
                f"{prefix}_dice_(weighted_avg)", weighted_avg.item(), batch_size=bs
            )
            self.log(f"{prefix}_dice_(macro_avg)", metric.mean().item(), batch_size=bs)

            for i, class_metric in enumerate(metric.detach().cpu()):
                if i == 0:  # NOTE: Skips background class.
                    continue
                self.log(
                    f"{prefix}_dice_class_{i}",
                    class_metric.item(),
                    batch_size=bs,
                )

    @override
    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters())
        if isinstance(self.scheduler, str):
            if self.scheduler == "gradual_warmup_scheduler":
                scheduler = {
                    "scheduler": LightningGradualWarmupScheduler(
                        optimizer,
                        multiplier=self.multiplier,
                        total_epoch=self.total_epochs,
                        T_max=50,
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss",
                    "strict": True,
                }
            else:
                raise NotImplementedError(
                    f"Scheduler of type {self.scheduler} not implemented"
                )
        else:
            scheduler = self.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_exception(self, exception: BaseException):
        raise exception


class CineDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/train_val-20240905T025601Z-001/train_val/",
        test_dir: str = "data/test-20240905T012341Z-001/test/",
        indices_dir: str = "data/indices/",
        batch_size: int = BATCH_SIZE_TRAIN,
        frames: int = NUM_FRAMES,
        select_frame_method: Literal["consecutive", "specific"] = "specific",
    ):
        """Datamodule for the Cine dataset for PyTorch Lightning compatibility.

        Args:
            data_dir: Path to train data directory containing Cine and masks
            subdirectories.
            test_dir: Path to test data directory containing Cine and masks
            subdirectories.
            indices_dir: Path to directory containing `train_indices.pkl` and
            `val_indices.pkl`.
            batch_size: Minibatch sizes for the DataLoader.
            frames: Number of frames from the original dataset to use.
            select_frame_method: How frames < 30 are selected for training.
        """
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.indices_dir = indices_dir
        self.batch_size = batch_size
        self.frames = frames
        self.select_frame_method = select_frame_method

    def setup(self, stage):
        indices_dir = os.path.join(os.getcwd(), self.indices_dir)

        trainval_img_dir = os.path.join(os.getcwd(), self.data_dir, "Cine")
        trainval_mask_dir = os.path.join(os.getcwd(), self.data_dir, "masks")
        transforms_img = Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        transforms_mask = Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        trainval_dataset = CineDataset(
            trainval_img_dir,
            trainval_mask_dir,
            indices_dir,
            frames=self.frames,
            select_frame_method=self.select_frame_method,
            transform_1=transforms_img,
            transform_2=transforms_mask,
        )
        assert len(trainval_dataset) > 0, "combined train/val set is empty"

        assert (idx := max(trainval_dataset.train_idxs)) < len(
            trainval_dataset
        ), f"Malformed training indices: {idx} for dataset of len: {len(trainval_dataset)}"

        assert (idx := max(trainval_dataset.valid_idxs)) < len(
            trainval_dataset
        ), f"Malformed training indices: {idx} for dataset of len: {len(trainval_dataset)}"

        train_set, valid_set = get_trainval_data_subsets(trainval_dataset)

        test_img_dir = os.path.join(os.getcwd(), self.test_dir, "Cine")
        test_mask_dir = os.path.join(os.getcwd(), self.test_dir, "masks")

        test_dataset = CineDataset(
            test_img_dir,
            test_mask_dir,
            indices_dir,
            frames=self.frames,
            select_frame_method=self.select_frame_method,
            transform_1=transforms_img,
            transform_2=transforms_mask,
            mode="test",
        )

        self.train = train_set
        self.val = valid_set
        self.test = test_dataset

    def on_exception(self, exception):
        raise exception

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8,
            drop_last=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8,
            persistent_workers=True,
        )


class TwoPlusOneCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_optimizer_args(AdamW)
        parser.add_lr_scheduler_args(LightningGradualWarmupScheduler)
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.add_class_arguments(TensorBoardLogger, "tensorboard")
        parser.add_argument("--version", type=Union[str, None], default=None)
        parser.link_arguments("version", "tensorboard.version")
        parser.link_arguments("tensorboard", "trainer.logger", apply_on="instantiate")

        parser.set_defaults(
            {
                "trainer.max_epochs": 50,
                "model.encoder_name": "resnet50",
                "model.encoder_weights": "imagenet",
                "model.in_channels": 3,
                "model.classes": 4,
                "model_checkpoint.monitor": "val_loss",
                "model_checkpoint.dirpath": os.path.join(
                    os.getcwd(), "checkpoints/two-plus-one/"
                ),
                "model_checkpoint.save_last": True,
                "model_checkpoint.save_weights_only": True,
                "model_checkpoint.save_top_k": 1,
                "tensorboard.save_dir": os.path.join(
                    os.getcwd(), "checkpoints/two-plus-one/"
                ),
            }
        )


if __name__ == "__main__":
    cli = TwoPlusOneCLI(
        UnetLightning,
        CineDataModule,
        # save_config_kwargs={
        #     "config_filename": "config.yaml",
        #     "multifile": True,
        # },
        save_config_callback=None,
        auto_configure_optimizers=False,
    )
