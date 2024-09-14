# -*- coding: utf-8 -*-
"""Two-plus-one architecture training script."""
from __future__ import annotations

import os
from typing import Any, Literal, OrderedDict, Union, override

import lightning as L
import segmentation_models_pytorch as smp
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from torch import nn
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Metric
from torchmetrics.segmentation import GeneralizedDiceScore
from torchvision.transforms import v2
from torchvision.transforms.transforms import Compose
from torchvision.utils import draw_segmentation_masks

from dataset.dataset import TwoPlusOneDataset, get_trainval_data_subsets
from metrics.dice import GeneralizedDiceScoreVariant
from models.two_plus_one import Unet
from utils import utils
from utils.utils import (
    ClassificationType,
    InverseNormalize,
    LightningGradualWarmupScheduler,
)

BATCH_SIZE_TRAIN = 4  # Default batch size for training.
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
NUM_FRAMES = 5  # Default number of frames.
torch.set_float32_matmul_precision("medium")


class UnetLightning(L.LightningModule):
    def __init__(
        self,
        metric: Metric | None = None,
        loss: nn.Module | str | None = None,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        num_frames: int = 5,
        weights_from_ckpt_path: str | None = None,
        optimizer: Optimizer | str = "adamw",
        optimizer_kwargs: dict[str, Any] = {},
        scheduler: LRScheduler | str = "gradual_warmup_scheduler",
        scheduler_kwargs: dict[str, Any] = {},
        multiplier: int = 2,
        total_epochs: int = 50,
        alpha: float = 1.0,
        _beta: float = 0.0,
        learning_rate: float = 1e-4,
        dl_classification_mode: ClassificationType = ClassificationType.MULTICLASS_MODE,
        eval_classification_mode: ClassificationType = ClassificationType.MULTICLASS_MODE,
    ):
        """A LightningModule wrapper for the modified Unet for the two-plus-one
        architecture.

        Args:
            metric: The metric to use for evaluation.
            loss: The loss function to use for training.
            encoder_name: The encoder name to use for the Unet.
            encoder_depth: The depth of the encoder.
            encoder_weights: The weights to use for the encoder.
            in_channels: The number of input channels.
            classes: The number of classes.
            num_frames: The number of frames to use.
            weights_from_ckpt_path: The path to the checkpoint to load weights from.
            optimizer: The optimizer to use.
            optimizer_kwargs: The optimizer keyword arguments.
            scheduler: The learning rate scheduler to use.
            scheduler_kwargs: The scheduler keyword arguments.
            multiplier: The multiplier for the learning rate.
            total_epochs: The total number of epochs.
            alpha: The alpha value for the loss function.
            _beta: The beta value for the loss function (Unused).
            learning_rate: The learning rate.
            dl_classification_mode: The classification mode for the dataloader.
            eval_classification_mode: The classification mode for evaluation.

        Raises:
            NotImplementedError: If the loss type is not implemented.
            RuntimeError: If the checkpoint is not loaded correctly.
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
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs

        # Sets loss if it's a string
        if isinstance(loss, str):
            match loss:
                case "cross_entropy":
                    class_weights = torch.Tensor(
                        [
                            0.00018531001957368073,
                            0.015518576429048081,
                            0.058786240529692384,
                            0.925509873021686,
                        ],
                    ).to(DEVICE)
                    self.loss = nn.CrossEntropyLoss(weight=class_weights)
                case "focal":
                    self.loss = FocalLoss("multiclass", normalized=True)
                case _:
                    raise NotImplementedError(
                        f"Loss type of {loss} is not implemented!"
                    )
        # Otherwise, set if nn.Module
        else:
            self.loss = (
                loss
                if isinstance(loss, nn.Module)
                else (
                    DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
                    if dl_classification_mode == ClassificationType.MULTILABEL_MODE
                    else DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
                )
            )

        # Set metric if none
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
        self.dl_classification_mode = dl_classification_mode
        self.eval_classification_mode = eval_classification_mode

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
                    self.model.load_state_dict(  # pyright: ignore[reportAttributeAccessIssue]
                        new_state_dict
                    )
                except RuntimeError as e:
                    raise e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # pyright: ignore[reportCallIssue]

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
            masks_proba: torch.Tensor = self.model(
                images_input
            )  # pyright: ignore[reportCallIssue]

        if self.dl_classification_mode == ClassificationType.MULTILABEL_MODE:
            # GUARD: Check that the sizes match.
            assert (
                masks_proba.size() == masks.size()
            ), f"Output of shape {masks_proba.shape} != target shape: {masks.shape}"

        # HACK: This ensures that the dimensions to the loss function are correct.
        if isinstance(self.loss, nn.CrossEntropyLoss) or isinstance(
            self.loss, FocalLoss
        ):
            loss_seg = self.alpha * self.loss(masks_proba, masks.squeeze(dim=1))
        else:
            loss_seg = self.alpha * self.loss(masks_proba, masks)
        loss_all = loss_seg
        self.log(f"train_loss", loss_all.item(), batch_size=bs, on_epoch=True)

        if isinstance(self.metric, GeneralizedDiceScore):
            masks_preds, masks_one_hot = utils.shared_metric_calculation(
                self, images, masks, masks_proba, "train"
            )

            self._shared_image_logging(
                batch_idx, images, masks_one_hot, masks_preds, "train", 25
            )
            self.train()

        return loss_all

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, str], batch_idx: int):
        self._shared_eval(batch, batch_idx, "test")

    @torch.no_grad()
    def _shared_image_logging(
        self,
        batch_idx: int,
        images: torch.Tensor,
        masks: torch.Tensor,
        masks_preds: torch.Tensor,
        prefix: str,
        every_interval: int = 10,
    ):
        """Logs the images to tensorboard.

        Args:
            batch_idx: The batch index.
            images: The input images.
            masks: The ground truth masks.
            masks_preds: The predicted masks.
            prefix: The runtime mode (train, val, test).
            every_interval: The interval to log images.

        Returns:
            None.

        Raises:
            AssertionError: If the logger is not detected or is not an instance of
            TensorboardLogger.
            ValueError: If any of `images`, `masks`, or `masks_preds` are malformed.
        """
        assert self.logger is not None, "No logger detected!"
        assert isinstance(
            self.logger, TensorBoardLogger
        ), f"Logger is not an instance of TensorboardLogger, but is of type {type(self.logger)}"

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
                global_step=batch_idx if prefix == "test" else self.global_step,
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
                global_step=batch_idx if prefix == "test" else self.global_step,
            )

    @torch.no_grad()
    def _shared_eval(
        self, batch: tuple[torch.Tensor, torch.Tensor, str], batch_idx: int, prefix: str
    ):
        """Shared evaluation step for validation and test steps.

        Args:
            batch: The batch of images and masks.
            batch_idx: The batch index.
            prefix: The runtime mode (val, test).
        """
        self.eval()
        images, masks, _ = batch
        bs = images.shape[0] if len(images.shape) > 3 else 1
        images_input = images.permute(1, 0, 2, 3, 4)
        images_input = images_input.to(DEVICE, dtype=torch.float32)
        masks = masks.to(DEVICE).long()
        masks_proba: torch.Tensor = self.model(
            images_input
        )  # pyright: ignore[reportCallIssue]

        if self.dl_classification_mode == ClassificationType.MULTILABEL_MODE:
            # GUARD: Check that the sizes match.
            assert (
                masks_proba.size() == masks.size()
            ), f"Output of shape {masks_proba.shape} != target shape: {masks.shape}"

        # HACK: This ensures that the dimensions to the loss function are correct.
        if isinstance(self.loss, nn.CrossEntropyLoss) or isinstance(
            self.loss, FocalLoss
        ):
            loss_seg = self.alpha * self.loss(masks_proba, masks.squeeze(dim=1))
        else:
            loss_seg = self.alpha * self.loss(masks_proba, masks)

        loss_all = loss_seg
        self.log(
            f"{prefix}_loss",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"hp_metric",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
        )

        if isinstance(self.metric, GeneralizedDiceScore):
            masks_preds, masks_one_hot = utils.shared_metric_calculation(
                self, images, masks, masks_proba, prefix
            )

            self._shared_image_logging(
                batch_idx, images, masks_one_hot, masks_preds, prefix, 10
            )

    @override
    def configure_optimizers(self):  # pyright: ignore[reportIncompatibleMethodOverride]
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
        num_workers: int = 8,
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
            classification_mode: The classification mode for the dataloader.
            num_workers: The number of workers for the DataLoader.
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
        self.num_workers = num_workers

    def setup(self, stage):
        indices_dir = os.path.join(os.getcwd(), self.indices_dir)

        trainval_img_dir = os.path.join(os.getcwd(), self.data_dir, "Cine")
        trainval_mask_dir = os.path.join(os.getcwd(), self.data_dir, "masks")
        transforms_img = Compose(
            [
                v2.ToImage(),
                v2.Resize(224, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        transforms_mask = Compose(
            [
                v2.ToImage(),
                v2.Resize(224, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        trainval_dataset = TwoPlusOneDataset(
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

        test_dataset = TwoPlusOneDataset(
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
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )


class TwoPlusOneCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        if (subcommand := getattr(self, "subcommand")) is not None:
            if (config := self.config.get(subcommand)) is not None:
                if (version := config.get("version")) is not None:
                    name = utils.get_last_checkpoint_filename(version)
                    ModelCheckpoint.CHECKPOINT_NAME_LAST = (  # pyright: ignore[reportAttributeAccessIssue]
                        name
                    )

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_optimizer_args(AdamW)
        parser.add_lr_scheduler_args(LightningGradualWarmupScheduler)
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.add_lightning_class_args(
            ModelCheckpoint, "model_checkpoint_dice_weighted"
        )
        parser.add_class_arguments(TensorBoardLogger, "tensorboard")
        parser.add_argument(
            "--version", type=Union[str, None], default=None, help="Experiment name"
        )
        parser.link_arguments("tensorboard", "trainer.logger", apply_on="instantiate")
        parser.link_arguments("model.num_frames", "data.frames")

        # Sets the checkpoint filename if version is provided.
        parser.link_arguments(
            "version",
            "model_checkpoint.filename",
            compute_fn=utils.get_checkpoint_filename,
        )
        parser.link_arguments(
            "version",
            "model_checkpoint_dice_weighted.filename",
            compute_fn=utils.get_best_weighted_avg_dice_filename,
        )
        parser.link_arguments("version", "tensorboard.name")

        # Adds the classification mode argument
        parser.add_argument("--dl_classification_mode", type=str)
        parser.add_argument("--eval_classification_mode", type=str)
        parser.link_arguments(
            "dl_classification_mode",
            "model.dl_classification_mode",
            compute_fn=utils.get_classification_mode,
        )
        parser.link_arguments(
            "eval_classification_mode",
            "model.eval_classification_mode",
            compute_fn=utils.get_classification_mode,
        )
        parser.link_arguments(
            "dl_classification_mode",
            "data.classification_mode",
            compute_fn=utils.get_classification_mode,
        )

        parser.set_defaults(
            {
                "dl_classification_mode": "MULTICLASS_MODE",
                "eval_classification_mode": "MULTILABEL_MODE",
                "trainer.max_epochs": 50,
                "model.encoder_name": "resnet50",
                "model.encoder_weights": "imagenet",
                "model.in_channels": 3,
                "model.classes": 4,
                "model_checkpoint.monitor": "val_loss",
                "model_checkpoint.save_last": True,
                "model_checkpoint.save_weights_only": True,
                "model_checkpoint.save_top_k": 1,
                "model_checkpoint_dice_weighted.monitor": "val_dice_(weighted_avg)",
                "model_checkpoint_dice_weighted.save_top_k": 1,
                "model_checkpoint_dice_weighted.save_weights_only": True,
                "model_checkpoint_dice_weighted.save_last": False,
                "model_checkpoint_dice_weighted.mode": "max",
                "tensorboard.save_dir": os.path.join(
                    os.getcwd(), "checkpoints/two-plus-one/lightning_logs"
                ),
                "tensorboard.log_graph": False,
            }
        )


if __name__ == "__main__":
    cli = TwoPlusOneCLI(
        UnetLightning,
        CineDataModule,
        save_config_callback=None,
        auto_configure_optimizers=False,
    )
