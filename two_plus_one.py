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
)
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from metrics.dice import GeneralizedDiceScoreVariant
from models.two_plus_one import Unet
from segmentation_models_pytorch.losses.dice import DiceLoss
from torch import nn
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Metric
from torchmetrics.segmentation import GeneralizedDiceScore
from torchvision.transforms import v2
from torchvision.transforms.transforms import Compose
from torchvision.utils import draw_segmentation_masks
from utils import utils
from utils.utils import ClassificationType, InverseNormalize
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
        optimizer: str = "adamw",
        scheduler: LRSchedulerCallable | str = "gradual_warmup_scheduler",
        multiplier: int = 2,
        total_epochs: int = 50,
        alpha: float = 1.0,
        _beta: float = 0.0,
        learning_rate: float = 1e-4,
        classification_mode: ClassificationType = ClassificationType.MULTICLASS_MODE,
    ):
        """A LightningModule wrapper for the modified Unet for the two-plus-one
        architecture.

        Args:
            metric: Optional, metric object for tracking performance. Defaults to None.
            loss: Optional, loss/criterion callable. Defaults to None.
            encoder_name: Optional, encoder/Backbone model name for the Unet. Defaults
                to "resnet34"
            encoder_depth: Optional,number of encoder blocks in the Unet. Defaults to 5.
            encoder_weights: Optional, weights for the encoder/backbone model. Defaults
                to "imagenet"
            in_channels: Optional, number of channels in input. Defaults to 3.
            classes: Optional, number of classes for output. Defaults to 1.
            num_frames: Optional, number of frames to use from dataset. Defaults to 5.
            weights_from_ckpt_path: Optional, checkpoint path to load weights from.
                Defaults to None.
            optimizer: Optional, instantiable optimizer for model parameters. Defaults
                to "adamw"
            scheduler: Optional, instantiable LR Scheduler or string (for custom
                schedulers). Defaults to "gradual_warmup_scheduler".
            multiplier: Optional, hyperparameter for the GradualWarmupScheduler LR
                Scheduler. Defaults to 2.
            total_epochs: Optional, hyperparameter for the CosineAnnealingLR portion of
                the GradualWarmupScheduler. Defaults to 50.
            alpha: Optional, hyperparameter for loss scaling. Defaults to 1.0.
            _beta: (Unused) Loss scaler.
            learning_rate: Optional, starting learning rate. Defaults to 1e-4.
            classification_mode: Optional, classification task. Defaults to
                MULTICLASS_MODE.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["metric", "loss"])
        self.in_channels = in_channels
        self.classes = classes
        self.num_frames = num_frames
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
            loss
            if loss
            else (
                DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
                if classification_mode == ClassificationType.MULTILABEL_MODE
                else DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
            )
        )
        self.metric = (
            metric
            if metric
            else GeneralizedDiceScoreVariant(
                num_classes=classes,
                per_class=True,
                include_background=True,
                weight_type="linear",
            )
        )

        self.multiplier = multiplier
        self.total_epochs = total_epochs
        self.alpha = alpha
        self.de_transform = Compose(
            [InverseNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        )
        self.example_input_array = torch.randn(
            (self.num_frames, 2, 3, 224, 224), dtype=torch.float32
        ).to(DEVICE)

        self.learning_rate = learning_rate
        self.classification_mode = classification_mode

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
        bs = images.shape[0] if len(images.shape) > 3 else 1
        images_input = images.permute(1, 0, 2, 3, 4)
        images_input = images_input.to(DEVICE, dtype=torch.float32)
        masks = masks.to(DEVICE).long()

        with torch.autocast(device_type=self.device.type):
            # B x C x H x W
            masks_proba: torch.Tensor = self.model(images_input)

        if self.classification_mode == ClassificationType.MULTILABEL_MODE:
            # GUARD: Check that the sizes match.
            assert (
                masks_proba.size() == masks.size()
            ), f"Output of shape {masks_proba.shape} != target shape: {masks.shape}"

        loss_seg = self.alpha * self.loss(masks_proba, masks)
        loss_all = loss_seg
        self.log(f"train_loss", loss_all.item(), batch_size=bs, on_epoch=True)

        if isinstance(self.metric, GeneralizedDiceScore):
            masks_preds = utils.shared_metric_calculation(
                self, images, masks, masks_proba, "train"
            )

            self._shared_image_logging(
                batch_idx, images, masks, masks_preds, "train", 25
            )

        return loss_all

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, str], batch_idx: int):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_image_logging(
        self,
        batch_idx: int,
        images: torch.Tensor,
        masks: torch.Tensor,
        masks_preds: torch.Tensor,
        prefix: str,
        every_interval: int = 10,
    ):
        if batch_idx % every_interval == 0:
            # This adds images to the tensorboard.
            tensorboard_logger: SummaryWriter = self.logger.experiment
            inv_norm_img = self.de_transform(images).detach().cpu()
            pred_images_with_masks = [
                draw_segmentation_masks(
                    img,
                    masks=mask.bool(),
                    alpha=0.7,
                    colors=["black", "red", "blue", "green"],
                )
                # Get only the first frame of images.
                for img, mask in zip(
                    inv_norm_img[:, 0, :, :, :].detach().cpu(),
                    masks_preds.detach().cpu(),
                )
            ]
            tensorboard_logger.add_images(
                tag=f"{prefix}_pred_masks_{batch_idx}",
                img_tensor=torch.stack(tensors=pred_images_with_masks, dim=0)
                .detach()
                .cpu(),
            )
            gt_images_with_masks = [
                draw_segmentation_masks(
                    img,
                    masks=mask.bool(),
                    alpha=0.7,
                    colors=["black", "red", "blue", "green"],
                )
                # Get only the first frame of images.
                for img, mask in zip(
                    inv_norm_img[:, 0, :, :, :].detach().cpu(),
                    masks.detach().cpu(),
                )
            ]
            tensorboard_logger.add_images(
                tag=f"{prefix}_gt_masks_{batch_idx}",
                img_tensor=torch.stack(tensors=gt_images_with_masks, dim=0)
                .detach()
                .cpu(),
            )

    def _shared_eval(
        self, batch: tuple[torch.Tensor, torch.Tensor, str], batch_idx: int, prefix: str
    ):
        images, masks, _ = batch
        bs = images.shape[0] if len(images.shape) > 3 else 1
        images = images.permute(1, 0, 2, 3, 4)
        images = images.to(DEVICE, dtype=torch.float32)
        masks = masks.to(DEVICE).long()
        masks_proba: torch.Tensor = self.model(images)  # BS x C x H x W

        if self.classification_mode == ClassificationType.MULTILABEL_MODE:
            # GUARD: Check that the sizes match.
            assert (
                masks_proba.size() == masks.size()
            ), f"Output of shape {masks_proba.shape} != target shape: {masks.shape}"

        loss_seg = self.alpha * self.loss(masks_proba, masks)
        loss_all = loss_seg
        self.log(f"{prefix}_loss", loss_all.detach().cpu().item(), batch_size=bs)

        if isinstance(self.metric, GeneralizedDiceScore):
            masks_preds = utils.shared_metric_calculation(
                self, images, masks, masks_proba, prefix
            )

            self._shared_image_logging(
                batch_idx, images, masks, masks_preds, prefix, 10
            )

    @override
    def configure_optimizers(self):
        return utils.configure_optimizers(self)

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
        classification_mode: ClassificationType = ClassificationType.MULTICLASS_MODE,
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
        self.classification_mode = classification_mode

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
            classification_mode=self.classification_mode,
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
            classification_mode=self.classification_mode,
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
        parser.link_arguments("model.num_frames", "data.frames")

        parser.add_argument("--classification_mode", type=str)
        parser.link_arguments(
            "classification_mode",
            "model.classification_mode",
            compute_fn=utils.get_classification_mode,
        )
        parser.link_arguments(
            "classification_mode",
            "data.classification_mode",
            compute_fn=utils.get_classification_mode,
        )

        # Sets the checkpoint filename if version is provided.
        parser.link_arguments(
            "version",
            "model_checkpoint.filename",
            compute_fn=utils.get_checkpoint_filename,
        )
        # # Gets the version name from the checkpoint path if provided.
        # parser.link_arguments(
        #     "model.weights_from_ckpt_path",
        #     "version",
        #     compute_fn=TwoPlusOneCLI.get_version_name,
        # )

        parser.set_defaults(
            {
                "classification_mode": "MULTICLASS_MODE",
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
                "tensorboard.log_graph": False,
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
