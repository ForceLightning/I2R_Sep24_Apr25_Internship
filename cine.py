# -*- coding: utf-8 -*-
"""Cine Baseline model training script."""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Literal, Union, override

import lightning as L
import segmentation_models_pytorch as smp
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from segmentation_models_pytorch.losses.dice import DiceLoss
from segmentation_models_pytorch.losses.focal import FocalLoss
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Metric, MetricCollection
from torchvision.transforms.v2 import Compose
from torchvision.utils import draw_segmentation_masks

from dataset.dataset import CineDataset, get_trainval_data_subsets
from metrics.dice import GeneralizedDiceScoreVariant
from metrics.logging import (
    setup_metrics,
    shared_metric_calculation,
    shared_metric_logging_epoch_end,
)
from utils import utils
from utils.utils import (
    ClassificationMode,
    InverseNormalize,
    LoadingMode,
    get_transforms,
)

BATCH_SIZE_TRAIN = 8  # Default batch size
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.set_float32_matmul_precision("medium")


class LightningUnetWrapper(L.LightningModule):
    def __init__(
        self,
        metric: Metric | None = None,
        loss: nn.Module | str | None = None,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        in_channels: int = 90,
        classes: int = 4,
        weights_from_ckpt_path: str | None = None,
        optimizer: Optimizer | str = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: LRScheduler | str = "gradual_warmup_scheduler",
        scheduler_kwargs: dict[str, Any] | None = None,
        multiplier: int = 2,
        total_epochs: int = 50,
        alpha: float = 1.0,
        _beta: float = 0.0,
        learning_rate: float = 1e-4,
        dl_classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        eval_classification_mode: ClassificationMode = ClassificationMode.MULTILABEL_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
        dump_memory_snapshot: bool = False,
    ):
        """Wrapper for the UNet model.

        Args:
            metric: Metric to use for evaluation.
            loss: Loss function to use for training.
            encoder_name: Name of the encoder to use.
            encoder_depth: The depth of the encoder.
            encoder_weights: Weights to use for the encoder.
            in_channels: Number of input channels.
            classes: Number of classes.
            weights_from_ckpt_path: Path to the checkpoint to load weights from.
            optimizer: Optimizer to use.
            optimizer_kwargs: Optimizer keyword arguments.
            scheduler: Learning rate scheduler to use.
            scheduler_kwargs: Learning rate scheduler keyword arguments.
            multiplier: Multiplier for the learning rate.
            total_epochs: Total number of epochs to train.
            alpha: Alpha value for the loss function.
            _beta: Beta value for the loss function.
            learning_rate: Learning rate for the optimizer.
            dl_classification_mode: Classification mode for the dataloader.
            eval_classification_mode: Classification mode for evaluation.
            loading_mode: Image loading mode.
            dump_memory_snapshot: Whether to dump a memory snapshot after training.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["loss"])
        self.dump_memory_snapshot = dump_memory_snapshot

        # Trace memory usage
        if self.dump_memory_snapshot:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python"
            )

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}

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
                    if dl_classification_mode == ClassificationMode.MULTILABEL_MODE
                    else DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
                )
            )

        # Sets metric if None.
        self.metrics = {}
        setup_metrics(self, metric, classes)

        self.multiplier = multiplier
        self.total_epochs = total_epochs
        self.alpha = alpha
        self.de_transform = (
            Compose(
                [
                    InverseNormalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    )
                ]
            )
            if loading_mode == LoadingMode.RGB
            else Compose([InverseNormalize(mean=[0.449], std=[0.226])])
        )
        self.example_input_array = torch.randn(
            (2, in_channels, 224, 224), dtype=torch.float32
        ).to(DEVICE)

        self.learning_rate = learning_rate
        self.dl_classification_mode = dl_classification_mode
        self.eval_classification_mode = eval_classification_mode

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

        self.loading_mode = loading_mode

    def on_train_start(self):
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.log_hyperparams(
                self.hparams,  # pyright: ignore[reportArgumentType]
                {
                    "hp/val_loss": 0,
                    "hp/val/dice_macro_avg": 0,
                    "hp/val/dice_macro_class_2_3": 0,
                    "hp/val/dice_weighted_avg": 0,
                    "hp/val/dice_weighted_class_2_3": 0,
                },
            )

    def on_train_end(self) -> None:
        if self.dump_memory_snapshot:
            torch.cuda.memory._dump_snapshot("unet_snapshot.pickle")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # pyright: ignore[reportCallIssue]

    def on_train_epoch_end(self) -> None:
        shared_metric_logging_epoch_end(self, "train")

    def on_validation_epoch_end(self) -> None:
        shared_metric_logging_epoch_end(self, "val")

    def on_test_epoch_end(self) -> None:
        shared_metric_logging_epoch_end(self, "test")

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ) -> torch.Tensor:
        images, masks, _ = batch
        bs: int = images.shape[0] if len(images.shape) > 3 else 1
        images = images.to(DEVICE, dtype=torch.float32)
        masks = masks.to(DEVICE).long()

        with torch.autocast(device_type=self.device.type):
            masks_proba: torch.Tensor = self.model(
                images
            )  # pyright: ignore[reportCallIssue]

            if self.dl_classification_mode == ClassificationMode.MULTILABEL_MODE:
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
            "loss/train",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"loss/train/{self.loss.__class__.__name__.lower()}",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
        )

        if isinstance(self.metrics["train"], GeneralizedDiceScoreVariant) or isinstance(
            self.metrics["train"], MetricCollection
        ):
            masks_preds, masks_one_hot = shared_metric_calculation(
                self, masks, masks_proba, "train"
            )

            if isinstance(self.logger, TensorBoardLogger):
                self._shared_image_logging(
                    batch_idx,
                    images.detach().cpu(),
                    masks_one_hot.detach().cpu(),
                    masks_preds.detach().cpu(),
                    "train",
                    25,
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

    @torch.no_grad()
    def _shared_eval(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        prefix: Literal["val", "test"],
    ):
        """Shared evaluation step for validation and test.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.
            prefix: Prefix for the logger.
        """
        images, masks, _ = batch
        images = images.to(DEVICE, dtype=torch.float32)  # BS x TS x C x H x W
        bs = images.shape[0] if len(images.shape) > 3 else 1
        masks = masks.to(DEVICE).long()
        masks_proba: torch.Tensor = self.model(
            images
        )  # pyright: ignore[reportCallIssue]

        if self.dl_classification_mode == ClassificationMode.MULTILABEL_MODE:
            # GUARD: Check that the sizes match.
            assert (
                masks_proba.size() == masks.size()
            ), f"Output of shape {masks_proba.shape} != target shape: {masks.shape}"

        loss_seg = self.alpha * self.loss(masks_proba, masks)
        loss_all = loss_seg
        self.log(
            f"loss/{prefix}",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"loss/{prefix}/{self.loss.__class__.__name__.lower()}",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
        )
        self.log(
            f"hp/{prefix}_loss",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
        )

        if isinstance(self.metrics[prefix], GeneralizedDiceScoreVariant) or isinstance(
            self.metrics[prefix], MetricCollection
        ):
            masks_preds, masks_one_hot = shared_metric_calculation(
                self, masks, masks_proba, prefix
            )

            if isinstance(self.logger, TensorBoardLogger):
                self._shared_image_logging(
                    batch_idx,
                    images.detach().cpu(),
                    masks_one_hot.detach().cpu(),
                    masks_preds.detach().cpu(),
                    prefix,
                    10,
                )

    @torch.no_grad()
    def _shared_image_logging(
        self,
        batch_idx: int,
        images: torch.Tensor,
        masks_one_hot: torch.Tensor,
        masks_preds: torch.Tensor,
        prefix: str,
        every_interval: int = 10,
    ):
        """Logs images to tensorboard.

        Args:
            batch_idx: Index of the batch.
            images: Images to log.
            masks_one_hot: Ground truth masks.
            masks_preds: Predicted masks.
            prefix: Prefix for the logger.
            every_interval: Interval to log images

        Returns:
            None

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

            # NOTE: This will adapt based on the color mode of the images
            if self.loading_mode == LoadingMode.RGB:
                inv_norm_img = self.de_transform(images[:, :3, :, :]).detach().cpu()
            else:
                image = (
                    images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1).detach().cpu()
                )
                inv_norm_img = self.de_transform(image).detach().cpu()

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
                    strict=True,
                )
            ]
            tensorboard_logger.add_images(
                tag=f"{prefix}/pred_masks_{batch_idx}",
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
                    inv_norm_img[:, 0:3, :, :].detach().cpu(),
                    masks_one_hot.detach().cpu(),
                    strict=True,
                )
            ]
            tensorboard_logger.add_images(
                tag=f"{prefix}/gt_masks_{batch_idx}",
                img_tensor=torch.stack(tensors=gt_images_with_masks, dim=0)
                .detach()
                .cpu(),
                global_step=batch_idx if prefix == "test" else self.global_step,
            )

    @override
    def configure_optimizers(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return utils.configure_optimizers(self)


class CineBaselineDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/train_val/",
        test_dir: str = "data/test/",
        indices_dir: str = "data/indices/",
        batch_size: int = BATCH_SIZE_TRAIN,
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        num_workers: int = 8,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
        augment: bool = False,
    ):
        """DataModule for the Cine baseline implementation.

        Args:
            data_dir: Path to the directory containing the training and validation data.
            test_dir: Path to the directory containing the test data.
            indices_dir: Path to the directory containing the indices.
            batch_size: Batch size for the data loader.
            classification_mode: Classification mode for the data loader.
            num_workers: Number of workers for the data loader.
            loading_mode: Determines the cv2.imread flags for the images.
            combine_train_val: Whether to combine train/val sets.
        """
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.indices_dir = indices_dir
        self.batch_size = batch_size
        self.classification_mode = classification_mode
        self.num_workers = num_workers
        self.loading_mode = loading_mode
        self.combine_train_val = combine_train_val
        self.augment = augment

    @override
    def setup(self, stage):
        indices_dir = os.path.join(os.getcwd(), self.indices_dir)

        trainval_img_dir = os.path.join(os.getcwd(), self.data_dir, "Cine")
        trainval_mask_dir = os.path.join(os.getcwd(), self.data_dir, "masks")

        transforms_img, transforms_mask, transforms_together = get_transforms(
            self.loading_mode, self.augment
        )

        trainval_dataset = CineDataset(
            trainval_img_dir,
            trainval_mask_dir,
            indices_dir,
            transform_img=transforms_img,
            transform_mask=transforms_mask,
            transform_together=transforms_together,
            classification_mode=self.classification_mode,
            loading_mode=self.loading_mode,
            combine_train_val=self.combine_train_val,
        )
        assert len(trainval_dataset) > 0, "combined train/val set is empty"

        test_img_dir = os.path.join(os.getcwd(), self.test_dir, "Cine")
        test_mask_dir = os.path.join(os.getcwd(), self.test_dir, "masks")

        test_dataset = CineDataset(
            test_img_dir,
            test_mask_dir,
            indices_dir,
            transform_img=transforms_img,
            transform_mask=transforms_mask,
            mode="test",
            classification_mode=self.classification_mode,
            loading_mode=self.loading_mode,
            combine_train_val=self.combine_train_val,
        )

        if self.combine_train_val:
            self.train = trainval_dataset
            self.val = test_dataset
            self.test = test_dataset
        else:
            assert (idx := max(trainval_dataset.train_idxs)) < len(
                trainval_dataset
            ), f"Malformed training indices: {idx} for dataset of len: {len(trainval_dataset)}"

            assert (idx := max(trainval_dataset.valid_idxs)) < len(
                trainval_dataset
            ), f"Malformed training indices: {idx} for dataset of len: {len(trainval_dataset)}"

            valid_dataset = CineDataset(
                trainval_img_dir,
                trainval_mask_dir,
                indices_dir,
                transform_img=transforms_img,
                transform_mask=transforms_mask,
                classification_mode=self.classification_mode,
                loading_mode=self.loading_mode,
                combine_train_val=self.combine_train_val,
            )

            train_set, valid_set = get_trainval_data_subsets(
                trainval_dataset, valid_dataset
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
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
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


class CineCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        if self.subcommand is not None:
            if (config := self.config.get(self.subcommand)) is not None:
                if (version := config.get("version")) is not None:
                    name = utils.get_last_checkpoint_filename(version)
                    ModelCheckpoint.CHECKPOINT_NAME_LAST = (  # pyright: ignore[reportAttributeAccessIssue]
                        name
                    )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.add_lightning_class_args(
            ModelCheckpoint, "model_checkpoint_dice_weighted"
        )
        parser.add_argument("--version", type=Union[str, None], default=None)

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
        parser.link_arguments("version", "trainer.logger.init_args.name")

        # Sets the image color loading mode
        parser.add_argument("--image_loading_mode", type=Union[str, None], default=None)
        parser.link_arguments(
            "image_loading_mode", "data.loading_mode", compute_fn=utils.get_loading_mode
        )
        parser.link_arguments(
            "image_loading_mode",
            "model.loading_mode",
            compute_fn=utils.get_loading_mode,
        )

        parser.link_arguments("trainer.max_epochs", "model.total_epochs")

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
                "image_loading_mode": "RGB",
                "dl_classification_mode": "MULTICLASS_MODE",
                "eval_classification_mode": "MULTILABEL_MODE",
                "trainer.max_epochs": 50,
                "model.encoder_name": "resnet50",
                "model.encoder_weights": "imagenet",
                "model.in_channels": 90,
                "model.classes": 4,
                "model_checkpoint.monitor": "loss/val",
                "model_checkpoint.save_last": True,
                "model_checkpoint.save_weights_only": True,
                "model_checkpoint.save_top_k": 1,
                "model_checkpoint.auto_insert_metric_name": False,
                "model_checkpoint_dice_weighted.monitor": "val/dice_weighted_avg",
                "model_checkpoint_dice_weighted.save_top_k": 1,
                "model_checkpoint_dice_weighted.save_weights_only": True,
                "model_checkpoint_dice_weighted.save_last": False,
                "model_checkpoint_dice_weighted.mode": "max",
                "model_checkpoint_dice_weighted.auto_insert_metric_name": False,
            }
        )


if __name__ == "__main__":
    cli = CineCLI(
        LightningUnetWrapper,
        CineBaselineDataModule,
        save_config_callback=None,
        auto_configure_optimizers=False,
        parser_kwargs={"fit": {"default_config_files": ["./configs/cine.yaml"]}},
    )
