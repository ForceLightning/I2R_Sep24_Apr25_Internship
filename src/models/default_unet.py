"""Contains the default U-Net implementation LightningModule wrapper."""

from __future__ import annotations

# Standard Library
import os
from collections import OrderedDict
from typing import Any, Literal, override

# Third-Party
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses.dice import DiceLoss
from segmentation_models_pytorch.losses.focal import FocalLoss

# PyTorch
import torch
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Metric, MetricCollection
from torchvision.transforms.v2 import Compose
from torchvision.utils import draw_segmentation_masks

# State-of-the-Art (SOTA) code
from thirdparty.TransUNet.networks.vit_seg_configs import get_r50_b16_config

# First party imports
from metrics.dice import GeneralizedDiceScoreVariant
from metrics.logging import (
    setup_metrics,
    shared_metric_calculation,
    shared_metric_logging_epoch_end,
)
from metrics.loss import WeightedDiceLoss
from models.common import CommonModelMixin
from models.transunet import TransUnet
from utils import utils
from utils.types import (
    INV_NORM_GREYSCALE_DEFAULT,
    INV_NORM_RGB_DEFAULT,
    ClassificationMode,
    DummyPredictMode,
    LoadingMode,
    MetricMode,
    ModelType,
)


class LightningUnetWrapper(CommonModelMixin):
    """LightningModule wrapper for U-Net model."""

    def __init__(
        self,
        batch_size: int,
        metric: Metric | None = None,
        num_frames: int = 30,
        loss: nn.Module | str | None = None,
        model_type: ModelType = ModelType.UNET,
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
        dummy_predict: DummyPredictMode = DummyPredictMode.NONE,
        metric_mode: MetricMode = MetricMode.INCLUDE_EMPTY_CLASS,
        metric_div_zero: float = 1.0,
    ):
        """Init the UNet model.

        Args:
            batch_size: Mini-batch size.
            metric: Metric to use for evaluation.
            num_frames: Number of frames to process.
            loss: Loss function to use for training.
            model_type: Model architecture to use.
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
            dummy_predict: Whether to predict ground truth masks for visualisation.
            metric_mode: Metric calculation mode.
            metric_div_zero: How to handle division by zero operations.

        """
        # Trace memory usage
        self.dump_memory_snapshot = dump_memory_snapshot
        if self.dump_memory_snapshot:
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks="python" if os.name == "nt" else "all",
            )

        super().__init__()
        self.save_hyperparameters(ignore=["loss"])
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.model_type = model_type
        self.dummy_predict = dummy_predict
        self.classes = classes

        match model_type:
            case ModelType.UNET:
                self.model = smp.Unet(  # pyright: ignore[reportAttributeAccessIssue]
                    encoder_name=encoder_name,
                    encoder_depth=encoder_depth,
                    encoder_weights=encoder_weights,
                    in_channels=in_channels,
                    classes=classes,
                )
            case ModelType.UNET_PLUS_PLUS:
                self.model = (
                    smp.UnetPlusPlus(  # pyright: ignore[reportAttributeAccessIssue]
                        encoder_name=encoder_name,
                        encoder_depth=encoder_depth,
                        encoder_weights=encoder_weights,
                        in_channels=in_channels,
                        classes=classes,
                    )
                )
            case ModelType.TRANS_UNET:
                config = get_r50_b16_config()
                config.patches.grid = (  # pyright: ignore
                    int(224 / config.patches.grid[0]),  # pyright: ignore
                    int(224 / config.patches.grid[1]),  # pyright: ignore
                )
                self.model = TransUnet(
                    config, img_size=224, num_classes=classes, in_channels=in_channels
                )

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}

        # Sets loss if it's a string
        if isinstance(loss, str):
            match loss:
                case "cross_entropy":
                    class_weights = Tensor(
                        [
                            0.000019931143,
                            0.001904109430,
                            0.010289336432,
                            0.987786622995,
                        ],
                    ).to(self.device.type)
                    self.loss = nn.CrossEntropyLoss(weight=class_weights)
                case "focal":
                    self.loss = FocalLoss("multiclass", normalized=True)
                case "weighted_dice":
                    class_weights = Tensor(
                        [
                            0.000019931143,
                            0.001904109430,
                            0.010289336432,
                            0.987786622995,
                        ],
                    ).to(self.device.type)
                    self.loss = (
                        WeightedDiceLoss(
                            classes, "multiclass", class_weights, from_logits=True
                        )
                        if dl_classification_mode == ClassificationMode.MULTICLASS_MODE
                        else WeightedDiceLoss(
                            classes, "multilabel", class_weights, from_logits=True
                        )
                    )
                case _:
                    raise NotImplementedError(
                        f"Loss type of {loss} is not implemented!"
                    )
        # Otherwise, set if nn.Module
        elif isinstance(loss, nn.Module):
            self.loss = loss
        else:
            match dl_classification_mode:
                case ClassificationMode.MULTICLASS_MODE:
                    self.loss = DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
                case ClassificationMode.MULTILABEL_MODE:
                    self.loss = DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
                case ClassificationMode.BINARY_CLASS_3_MODE:
                    self.loss = DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

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
        self.example_input_array = torch.randn(
            (self.batch_size, in_channels, 224, 224), dtype=torch.float32
        ).to(self.device.type)

        self.learning_rate = learning_rate
        self.dl_classification_mode = dl_classification_mode
        self.eval_classification_mode = eval_classification_mode

        # Sets metric if None.
        self.dice_metrics = {}
        self.other_metrics = {}
        setup_metrics(self, metric, classes, metric_mode, metric_div_zero)

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

    @override
    def on_train_start(self):
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.log_hyperparams(
                self.hparams,  # pyright: ignore[reportArgumentType]
                {
                    "hp/val_loss": 0,
                    "hp/val/dice_macro_avg": 0,
                    "hp/val/dice_weighted_avg": 0,
                    "hp/val/dice_macro_class_2_3": 0,
                    "hp/val/dice_weighted_class_2_3": 0,
                    "hp/val/dice_class_1": 0,
                    "hp/val/dice_class_2": 0,
                    "hp/val/dice_class_3": 0,
                },
            )

    @override
    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type=self.device.type):
            return self.model(x)  # pyright: ignore[reportCallIssue]

    @override
    def log_metrics(self, prefix) -> None:
        shared_metric_logging_epoch_end(self, prefix)

    def training_step(
        self, batch: tuple[Tensor, Tensor, str], batch_idx: int
    ) -> Tensor:
        """Forward pass for the model with dataloader batches.

        Args:
            batch: Batch of frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.

        Return:
            Training loss.

        Raises:
            AssertionError: Prediction shape and ground truth mask shapes are different.

        """
        images, masks, _ = batch
        bs: int = images.shape[0] if len(images.shape) > 3 else 1

        with torch.autocast(device_type=self.device.type):
            masks_proba: Tensor = self.forward(images)

            if (
                self.dl_classification_mode == ClassificationMode.MULTILABEL_MODE
                or self.dl_classification_mode == ClassificationMode.BINARY_CLASS_3_MODE
            ):
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
            sync_dist=True,
        )
        self.log(
            f"loss/train/{self.loss.__class__.__name__.lower()}",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
        )

        if isinstance(
            self.dice_metrics["train"], GeneralizedDiceScoreVariant
        ) or isinstance(self.dice_metrics["train"], MetricCollection):
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
        return loss_all

    @override
    def validation_step(self, batch: tuple[Tensor, Tensor, str], batch_idx: int):
        self._shared_eval(batch, batch_idx, "val")

    @override
    def test_step(self, batch: tuple[Tensor, Tensor, str], batch_idx: int) -> None:
        self._shared_eval(batch, batch_idx, "test")

    @torch.no_grad()
    def _shared_eval(
        self,
        batch: tuple[Tensor, Tensor, str],
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
        bs = images.shape[0] if len(images.shape) > 3 else 1
        images_input = images.to(self.device.type)
        masks = masks.to(self.device.type).long()
        masks_proba = self.model(images_input)  # pyright: ignore[reportCallIssue]

        if (
            self.dl_classification_mode == ClassificationMode.MULTILABEL_MODE
            or self.dl_classification_mode == ClassificationMode.BINARY_CLASS_3_MODE
        ):
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
            sync_dist=True,
        )
        self.log(
            f"loss/{prefix}/{self.loss.__class__.__name__.lower()}",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"hp/{prefix}_loss",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
        )

        if isinstance(
            self.dice_metrics[prefix], GeneralizedDiceScoreVariant
        ) or isinstance(self.dice_metrics[prefix], MetricCollection):
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
        images: Tensor,
        masks_one_hot: Tensor,
        masks_preds: Tensor,
        prefix: Literal["train", "val", "test"],
        every_interval: int = 10,
    ):
        """Log images to tensorboard.

        Args:
            batch_idx: Index of the batch.
            images: Images to log.
            masks_one_hot: Ground truth masks.
            masks_preds: Predicted masks.
            prefix: Prefix for the logger.
            every_interval: Interval to log images

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
        batch: tuple[Tensor, Tensor, str | list[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[Tensor, Tensor, str | list[str]]:
        """Forward pass for the model for one minibatch of a test epoch.

        Args:
            batch: Batch of frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.
            dataloader_idx: Index of the dataloader.

        Return:
            Mask predictions, original images, and filename.

        """
        self.eval()
        images, masks, fn = batch
        images_input = images.to(self.device.type)
        masks = masks.to(self.device.type).long()

        masks_preds: Tensor
        if self.dummy_predict == DummyPredictMode.GROUND_TRUTH:
            if self.eval_classification_mode == ClassificationMode.MULTICLASS_MODE:
                masks_preds = (
                    F.one_hot(masks, num_classes=4).permute(0, -1, 1, 2).bool()
                )
            else:
                masks_preds = masks.bool()
        elif self.dummy_predict == DummyPredictMode.BLANK:
            if self.eval_classification_mode == ClassificationMode.MULTICLASS_MODE:
                masks_preds = (
                    F.one_hot(torch.zeros_like(masks), num_classes=4)
                    .permute(0, -1, 1, 2)
                    .bool()
                )
            else:
                masks_preds = torch.zeros_like(masks).bool()
        else:
            assert isinstance(self.model, nn.Module)
            masks_proba: Tensor = self.model(images_input)

            if self.eval_classification_mode == ClassificationMode.MULTICLASS_MODE:
                masks_preds = masks_proba.argmax(dim=1)
                masks_preds = (
                    F.one_hot(masks_preds, num_classes=4).permute(0, -1, 1, 2).bool()
                )
            else:
                masks_preds = masks_proba > 0.5

        b, c, h, w = images.shape
        match self.loading_mode:
            case LoadingMode.RGB:
                reshaped_images = images.view(b, c // 3, 3, h, w)
            case LoadingMode.GREYSCALE:
                reshaped_images = images.view(b, c, 1, h, w)

        return masks_preds.detach().cpu(), reshaped_images.detach().cpu(), fn

    @override
    def configure_optimizers(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return utils.configure_optimizers(self)
