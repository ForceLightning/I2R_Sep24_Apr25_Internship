from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Literal, Union, override

import cv2
import lightning as L
import numpy as np
import segmentation_models_pytorch as smp
import torch
from cv2 import typing as cvt
from dataset.dataset import CineDataset, get_trainval_data_subsets
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
    LRSchedulerCallable,
    OptimizerCallable,
)
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from numpy import typing as npt
from segmentation_models_pytorch.losses.dice import DiceLoss
from torch import nn
from torch.nn import functional as F
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Dice, Metric
from torchmetrics.segmentation import GeneralizedDiceScore
from torchvision.transforms import v2
from torchvision.transforms.transforms import Compose
from torchvision.utils import draw_segmentation_masks
from two_plus_one import LightningGradualWarmupScheduler
from utils.utils import InverseNormalize

BATCH_SIZE_TRAIN = 4

BATCH_SIZE_VAL = 4
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
LEARNING_RATE = 1e-4
NUM_FRAMES = 5
SEED_CUS = 1  # RNG seed.
torch.set_float32_matmul_precision("medium")


class CineBaselineDataset(CineDataset):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        idxs_dir: str,
        transform_1: Compose,
        transform_2: Compose,
        batch_size: int = BATCH_SIZE_TRAIN,
        mode: Literal["train", "val", "test"] = "train",
    ) -> None:
        super(CineDataset).__init__()

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_list: list[str] = os.listdir(self.img_dir)
        self.mask_list: list[str] = os.listdir(self.mask_dir)

        self.transform_1 = transform_1
        self.transform_2 = transform_2

        self.train_idxs: list[int]
        self.valid_idxs: list[int]

        self.batch_size = batch_size

        if mode != "test":
            self.load_train_indices(
                os.path.join(idxs_dir, "train_indices.pkl"),
                os.path.join(idxs_dir, "val_indices.pkl"),
            )

    @override
    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor | npt.NDArray[np.floating[Any]], str]:
        # Define Cine file name
        img_name: str = self.img_list[index]
        mask_name: str = self.img_list[index].split(".")[0] + ".nii.png"

        img_tuple: tuple[bool, Sequence[cvt.MatLike]] = cv2.imreadmulti(
            os.path.join(self.img_dir, img_name), flags=cv2.IMREAD_COLOR
        )

        img_list = img_tuple[1]
        first_img = img_list[0]
        tuned = first_img / [255.0]
        tuned = cv2.resize(tuned, (224, 224))
        tuned = tuned.astype(np.float32)
        if self.transform_1:
            tuned = self.transform_1(tuned)
        else:
            tuned = torch.from_numpy(tuned)

        combined_imgs = tuned.permute(1, 2, 0)

        for i in range(len(img_list) - 1):
            img = img_list[i + 1]
            lab_img = img / [255.0]
            lab_img = cv2.resize(lab_img, (224, 224))
            lab_img = lab_img.astype(np.float32)
            if self.transform_1:
                lab_img = self.transform_1(lab_img)
            else:
                lab_img = torch.from_numpy(lab_img)

            combined_imgs = torch.dstack((combined_imgs, lab_img.permute(1, 2, 0)))

        mask = cv2.imread(os.path.join(self.mask_dir, mask_name))
        lab_mask = mask / [1.0]
        lab_mask = cv2.resize(lab_mask, (224, 224))
        lab_mask = lab_mask.astype(np.float32)
        lab_mask = lab_mask[:, :, 0]  # H x W
        if self.transform_2:
            lab_mask = self.transform_2(lab_mask)
        else:
            lab_mask = torch.from_numpy(lab_mask)

        combined_imgs = torch.swapaxes(combined_imgs, 0, 2)
        combined_cines = torch.flip(v2.functional.rotate(combined_imgs, 270), [2])

        return combined_cines, lab_mask, img_name


class LightningUnetWrapper(L.LightningModule):
    def __init__(
        self,
        metric: Metric | None = None,
        loss: nn.Module = DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True),
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 90,
        classes: int = 4,
        optimizer: OptimizerCallable = AdamW,
        scheduler: LRSchedulerCallable | str = "gradual_warmup_scheduler",
        multiplier: int = 2,
        total_epochs: int = 50,
        alpha: float = 1.0,
        _beta: float = 0.0,
        weights_from_ckpt_path: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss"])
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
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

        self.example_input_array = torch.randn(
            (2, in_channels, 224, 224), dtype=torch.float32
        ).to(DEVICE)

        self.de_transform = Compose(
            [InverseNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        )

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
        bs: int = images.shape[0] if len(images.shape) > 3 else 1
        images = images.to(DEVICE, dtype=torch.float32)
        masks = masks.to(DEVICE).long()

        with torch.autocast(device_type=self.device.type):
            masks_proba: torch.Tensor = self.model(images)

        assert (
            masks_proba.size() == masks.size()
        ), f"Output of shape {masks_proba.shape} != target shape {masks.shape}"

        loss_seg = self.alpha * self.loss(masks_proba, masks)
        loss_all = loss_seg
        self.log(
            f"train_loss",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            prog_bar=True,
        )

        if isinstance(self.metric, GeneralizedDiceScore):
            masks_preds = masks_proba > 0.5

            metric: torch.Tensor = self.metric(masks_preds, masks)
            self.log(
                f"train_dice_(macro_avg)",
                metric.mean().item(),
                batch_size=bs,
                on_epoch=True,
            )

            for i, class_metric in enumerate(metric.detach().cpu()):
                if i == 0:  # NOTE: Skips background class.
                    continue
                self.log(
                    f"train_dice_class_{i}",
                    class_metric.item(),
                    batch_size=bs,
                    on_epoch=True,
                )
            self._shared_image_logging(
                batch_idx, images, masks, masks_preds, "train", 20
            )
        return loss_all

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ) -> None:
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
                    inv_norm_img[:, 0:3, :, :].detach().cpu(),
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
                    inv_norm_img[:, 0:3, :, :].detach().cpu(),
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
        self,
        batch: tuple[torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        prefix: str,
    ):
        images, masks, _ = batch
        images = images.to(DEVICE, dtype=torch.float32)  # BS x TS x C x H x W
        bs = images.shape[0] if len(images.shape) > 3 else 1
        masks = masks.to(DEVICE).long()
        masks_proba: torch.Tensor = self.model(images)
        loss_seg = self.alpha * self.loss(masks_proba, masks)
        loss_all = loss_seg
        self.log(
            f"{prefix}_loss",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            prog_bar=True,
        )

        if isinstance(self.metric, GeneralizedDiceScore):
            masks_preds = masks_proba > 0.5

            metric = self.metric(masks_preds, masks)
            self.log(f"{prefix}_dice_(macro_avg)", metric.mean().item(), batch_size=bs)

            for i, class_metric in enumerate(metric.detach().cpu()):
                if i == 0:  # NOTE: Skips background class.
                    continue
                self.log(
                    f"{prefix}_dice_class_{i}",
                    class_metric.item(),
                    batch_size=bs,
                )
            self._shared_image_logging(batch_idx, images, masks, masks_preds, prefix)

    @override
    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters())
        if isinstance(self.scheduler, str):
            if self.scheduler == "gradual_warmup_scheduler":
                scheduler = {
                    "scheduler": LightningGradualWarmupScheduler(
                        optimizer,
                        multiplier=self.multiplier,
                        total_epoch=5,
                        T_max=self.total_epochs,
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


class CineBaselineDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/train_val-20240905T025601Z-001/train_val/",
        test_dir: str = "data/test-20240905T012341Z-001/test/",
        indices_dir: str = "data/indices/",
        batch_size: int = BATCH_SIZE_TRAIN,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.indices_dir = indices_dir
        self.batch_size = batch_size

    @override
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
        trainval_dataset = CineBaselineDataset(
            trainval_img_dir,
            trainval_mask_dir,
            indices_dir,
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

        test_dataset = CineBaselineDataset(
            test_img_dir,
            test_mask_dir,
            indices_dir,
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


class CineCLI(LightningCLI):
    @classmethod
    def get_checkpoint_filename(cls, version: str | None) -> str | None:
        if version:
            return version + "-epoch={epoch}-step={step}"
        else:
            return version

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(AdamW)
        parser.add_lr_scheduler_args(LightningGradualWarmupScheduler)
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.add_class_arguments(TensorBoardLogger, "tensorboard")
        parser.add_argument("--version", type=Union[str, None], default=None)
        parser.link_arguments("version", "tensorboard.version")
        parser.link_arguments("tensorboard", "trainer.logger", apply_on="instantiate")
        parser.link_arguments(
            "version",
            "model_checkpoint.filename",
            compute_fn=CineCLI.get_checkpoint_filename,
        )

        parser.set_defaults(
            {
                "trainer.max_epochs": 50,
                "model.encoder_name": "resnet50",
                "model.encoder_weights": "imagenet",
                "model.in_channels": 90,
                "model.classes": 4,
                "model_checkpoint.monitor": "val_loss",
                "model_checkpoint.dirpath": os.path.join(
                    os.getcwd(), "checkpoints/cine-baseline/"
                ),
                "model_checkpoint.save_last": True,
                "model_checkpoint.save_weights_only": True,
                "model_checkpoint.save_top_k": 1,
                "tensorboard.save_dir": os.path.join(
                    os.getcwd(), "checkpoints/cine-baseline/"
                ),
            }
        )


if __name__ == "__main__":
    cli = CineCLI(
        LightningUnetWrapper,
        CineBaselineDataModule,
        # save_config_kwargs={
        #     "config_filename": "config.yaml",
        #     "multifile": True,
        # },
        save_config_callback=None,
        auto_configure_optimizers=False,
    )
