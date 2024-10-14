# -*- coding: utf-8 -*-
"""Two-plus-one architecture training script."""
from __future__ import annotations

import os
from typing import Any, Literal, OrderedDict, override

import lightning as L
import segmentation_models_pytorch as smp
import torch
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Metric, MetricCollection
from torchvision.transforms.v2 import Compose
from torchvision.utils import draw_segmentation_masks

from cli.common import CommonCLI
from dataset.dataset import TwoPlusOneDataset, get_trainval_data_subsets
from metrics.dice import GeneralizedDiceScoreVariant
from metrics.logging import (
    setup_metrics,
    shared_metric_calculation,
    shared_metric_logging_epoch_end,
)
from models.two_plus_one import TwoPlusOneUnet
from utils import utils
from utils.utils import (
    INV_NORM_GREYSCALE_DEFAULT,
    INV_NORM_RGB_DEFAULT,
    ClassificationMode,
    LoadingMode,
)

BATCH_SIZE_TRAIN = 4  # Default batch size for training.
NUM_FRAMES = 5  # Default number of frames.
torch.set_float32_matmul_precision("medium")


class TwoPlusOneUnetLightning(L.LightningModule):
    """A LightningModule wrapper for the modified 2+1 U-Net architecture."""

    def __init__(
        self,
        batch_size: int,
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
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: LRScheduler | str = "gradual_warmup_scheduler",
        scheduler_kwargs: dict[str, Any] | None = None,
        multiplier: int = 2,
        total_epochs: int = 50,
        alpha: float = 1.0,
        _beta: float = 0.0,
        learning_rate: float = 1e-4,
        dl_classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        eval_classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
        dump_memory_snapshot: bool = False,
        flat_conv: bool = False,
        unet_activation: str | None = None,
    ):
        """Init the 2+1 U-Net LightningModule.

        Args:
            batch_size: Mini-batch size.
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
            multiplier: The multiplier for the learning rate to reach in the warmup.
            total_epochs: The total number of epochs.
            alpha: The alpha value for the loss function.
            _beta: The beta value for the loss function (Unused).
            learning_rate: The learning rate.
            dl_classification_mode: The classification mode for the dataloader.
            eval_classification_mode: The classification mode for evaluation.
            loading_mode: Image loading mode.
            dump_memory_snapshot: Whether to dump a memory snapshot after training.
            flat_conv: Whether to use a flat temporal convolutional layer.
            unet_activation: The activation function for the U-Net.

        Raises:
            NotImplementedError: If the loss type is not implemented.
            RuntimeError: If the checkpoint is not loaded correctly.

        """
        super().__init__()
        self.save_hyperparameters(ignore=["metric", "loss"])
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.classes = classes
        self.num_frames = num_frames
        self.dump_memory_snapshot = dump_memory_snapshot

        # Trace memory usage
        if self.dump_memory_snapshot:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python"
            )

        # PERF: The model can be `torch.compile()`'d but layout issues occur with
        # convolutional networks. See: https://github.com/pytorch/pytorch/issues/126585
        self.model = TwoPlusOneUnet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            num_frames=num_frames,
            flat_conv=flat_conv,
            activation=unet_activation,
        )
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}
        self.loading_mode = loading_mode

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
                    ).to(self.device.type)
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
                # If none
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
        self.de_transform = Compose(
            [
                (
                    INV_NORM_RGB_DEFAULT
                    if loading_mode == LoadingMode.RGB
                    else INV_NORM_GREYSCALE_DEFAULT
                )
            ]
        )
        # NOTE: This is to help with reproducibility
        with torch.random.fork_rng(devices=("cpu", "cuda:0")):
            self.example_input_array = torch.randn(
                (self.batch_size, self.num_frames, self.in_channels, 224, 224),
                dtype=torch.float32,
            ).to(self.device.type)

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

    def on_train_start(self):
        """Call at the beginning of training after sanity check."""
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.log_hyperparams(
                self.hparams,  # pyright: ignore[reportArgumentType]
                {
                    "hp/val_loss": 0,
                    "hp/val/dice_macro_avg": 0,
                    "hp/val/dice_macro_class_2_3": 0,
                    "hp/val/dice_weighted_avg": 0,
                    "hp/val/dice_weighted_class_2_3": 0,
                    "hp/val/dice_class_1": 0,
                    "hp/val/dice_class_2": 0,
                    "hp/val/dice_class_3": 0,
                },
            )

    @override
    def on_train_end(self) -> None:
        if self.dump_memory_snapshot:
            torch.cuda.memory._dump_snapshot("two_plus_one_snapshot.pickle")

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type):
            return self.model(x)  # pyright: ignore[reportCallIssue]

    @override
    def on_train_epoch_end(self) -> None:
        shared_metric_logging_epoch_end(self, "train")

    @override
    def on_validation_epoch_end(self) -> None:
        shared_metric_logging_epoch_end(self, "val")

    @override
    def on_test_epoch_end(self) -> None:
        shared_metric_logging_epoch_end(self, "test")

    @override
    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ) -> torch.Tensor:
        """Forward pass for the model with dataloader batches.

        Args:
            batch: Batch of frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.

        Return:
            torch.tensor: Training loss.

        Raises:
            AssertionError: Prediction shape and ground truth mask shapes are different.

        """
        images, masks, _ = batch
        bs = images.shape[0] if len(images.shape) > 3 else 1
        images_input = images.to(self.device.type, dtype=torch.float32)
        masks = masks.to(self.device.type).long()

        with torch.autocast(device_type=self.device.type):
            # B x C x H x W
            masks_proba: torch.Tensor = self.model(
                images_input
            )  # pyright: ignore[reportCallIssue] False positive

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
            "loss/train", loss_all.item(), batch_size=bs, on_epoch=True, prog_bar=True
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
                    10,
                )
            self.train()

        return loss_all

    @override
    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ):
        self._shared_eval(batch, batch_idx, "val")

    @override
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, str], batch_idx: int):
        self._shared_eval(batch, batch_idx, "test")

    @torch.no_grad()
    def _shared_eval(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        prefix: Literal["val", "test"],
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
        images_input = images.to(self.device.type, dtype=torch.float32)
        masks = masks.to(self.device.type).long()
        masks_proba: torch.Tensor = self.model(
            images_input
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
        prefix: Literal["train", "val", "test"],
        every_interval: int = 10,
    ):
        """Log the images to tensorboard.

        Args:
            batch_idx: The batch index.
            images: The input images.
            masks_one_hot: The ground truth masks.
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

            match prefix:
                case "val" | "test":
                    step = int(
                        sum(self.trainer.num_val_batches) * self.trainer.current_epoch
                        + batch_idx
                    )
                case _:
                    step = self.global_step

            # NOTE: This will adapt based on the color mode of the images
            if self.loading_mode == LoadingMode.RGB:
                inv_norm_img = self.de_transform(images).detach().cpu()
            else:
                image = (
                    images[:, :, 0, :, :]
                    .unsqueeze(2)
                    .repeat(1, 1, 3, 1, 1)
                    .detach()
                    .cpu()
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
                    inv_norm_img[:, 0, :, :, :].detach().cpu(),
                    masks_preds.detach().cpu(),
                    strict=True,
                )
            ]
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
                    masks_one_hot.detach().cpu(),
                    strict=True,
                )
            ]
            combined_images_with_masks = gt_images_with_masks + pred_images_with_masks

            tensorboard_logger.add_images(
                tag=f"{prefix}/preds",
                img_tensor=torch.stack(tensors=combined_images_with_masks, dim=0)
                .detach()
                .cpu(),
                global_step=step,
            )

    @override
    @torch.no_grad()
    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, str | list[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Forward pass for the model for one minibatch of a test epoch.

        Args:
            batch: Batch of frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.
            dataloader_idx: Index of the dataloader.

        Return:
            tuple[torch.tensor, torch.tensor, str]: Mask predictions, original images,
                and filename.

        """
        self.eval()
        images, masks, fn = batch
        images_input = images.to(self.device.type)
        masks = masks.to(self.device.type).long()

        masks_proba: torch.Tensor = self.model(
            images_input
        )  # pyright: ignore[reportCallIssue]

        if self.eval_classification_mode == ClassificationMode.MULTICLASS_MODE:
            masks_preds = masks_proba.argmax(dim=1)
            masks_preds = F.one_hot(masks_preds, num_classes=4).permute(0, -1, 1, 2)
        else:
            masks_preds = masks_proba > 0.5

        return masks_preds.detach().cpu(), images.detach().cpu(), fn

    @override
    def configure_optimizers(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return utils.configure_optimizers(self)


class TwoPlusOneDataModule(L.LightningDataModule):
    """Datamodule for the TwoPlusOne dataset for PyTorch Lightning compatibility."""

    def __init__(
        self,
        data_dir: str = "data/train_val/",
        test_dir: str = "data/test/",
        indices_dir: str = "data/indices/",
        batch_size: int = BATCH_SIZE_TRAIN,
        frames: int = NUM_FRAMES,
        select_frame_method: Literal["consecutive", "specific"] = "specific",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        num_workers: int = 8,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
        augment: bool = False,
    ):
        """Init the 2+1 datamodule.

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
            loading_mode: Determines the cv2.imread flags for the images.
            combine_train_val: Whether to combine train/val sets.
            augment: Whether to augment images and masks together.

        """
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.indices_dir = indices_dir
        self.batch_size = batch_size
        self.frames = frames
        self.select_frame_method: Literal["consecutive", "specific"] = (
            select_frame_method
        )
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

        # Get transforms for the CINE images, masks, and combined transforms.
        transforms_img, transforms_mask, transforms_together, transforms_resize = (
            utils.get_transforms(self.loading_mode, self.augment)
        )

        trainval_dataset = TwoPlusOneDataset(
            trainval_img_dir,
            trainval_mask_dir,
            indices_dir,
            frames=self.frames,
            select_frame_method=self.select_frame_method,
            transform_img=transforms_img,
            transform_mask=transforms_mask,
            transform_resize=transforms_resize,
            transform_together=transforms_together,
            classification_mode=self.classification_mode,
            loading_mode=self.loading_mode,
            combine_train_val=self.combine_train_val,
        )
        assert len(trainval_dataset) > 0, "combined train/val set is empty"

        test_img_dir = os.path.join(os.getcwd(), self.test_dir, "Cine")
        test_mask_dir = os.path.join(os.getcwd(), self.test_dir, "masks")

        test_dataset = TwoPlusOneDataset(
            test_img_dir,
            test_mask_dir,
            indices_dir,
            frames=self.frames,
            select_frame_method=self.select_frame_method,
            transform_img=transforms_img,
            transform_mask=transforms_mask,
            transform_resize=transforms_resize,
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

            valid_dataset = TwoPlusOneDataset(
                trainval_img_dir,
                trainval_mask_dir,
                indices_dir,
                frames=self.frames,
                select_frame_method=self.select_frame_method,
                transform_img=transforms_img,
                transform_mask=transforms_mask,
                transform_resize=transforms_resize,
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

    @override
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

    @override
    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    @override
    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    @override
    def predict_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=False,
        )


class TwoPlusOneCLI(CommonCLI):
    """CLI class for cine CMR 2+1 task."""

    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)

        defaults = self.default_arguments | {
            "image_loading_mode": "RGB",
            "dl_classification_mode": "MULTICLASS_MODE",
            "eval_classification_mode": "MULTILABEL_MODE",
            "trainer.max_epochs": 50,
            "model.encoder_name": "resnet50",
            "model.encoder_weights": "imagenet",
            "model.in_channels": 3,
            "model.classes": 4,
        }

        parser.set_defaults(defaults)


if __name__ == "__main__":
    cli = TwoPlusOneCLI(
        TwoPlusOneUnetLightning,
        TwoPlusOneDataModule,
        save_config_callback=None,
        auto_configure_optimizers=False,
        parser_kwargs={
            "fit": {"default_config_files": ["./configs/two_plus_one.yaml"]},
            "predict": {
                "default_config_files": [
                    "./configs/two_plus_one.yaml",
                    "./configs/predict.yaml",
                ]
            },
        },
    )
