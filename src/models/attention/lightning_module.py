# -*- coding: utf-8 -*-
"""LightningModule wrappers for U-Net with Attention mechanism on residual frames."""

from __future__ import annotations

from typing import Any, Literal, OrderedDict, override

import segmentation_models_pytorch as smp
import torch
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Metric, MetricCollection
from torchvision.transforms.v2 import Compose
from torchvision.utils import draw_segmentation_masks

from metrics.dice import GeneralizedDiceScoreVariant
from metrics.logging import (
    setup_metrics,
    shared_metric_calculation,
    shared_metric_logging_epoch_end,
)
from models.attention.model import REDUCE_TYPES
from models.attention.segmentation_model import (
    ResidualAttentionUnet,
    ResidualAttentionUnetPlusPlus,
)
from models.common import CommonModelMixin
from utils import utils
from utils.types import (
    INV_NORM_GREYSCALE_DEFAULT,
    INV_NORM_RGB_DEFAULT,
    ClassificationMode,
    LoadingMode,
    ModelType,
    ResidualMode,
)


class ResidualAttentionLightningModule(CommonModelMixin):
    """Attention mechanism-based U-Net."""

    def __init__(
        self,
        batch_size: int,
        metric: Metric | None = None,
        loss: nn.Module | str | None = None,
        model_type: ModelType = ModelType.UNET,
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
        residual_mode: ResidualMode = ResidualMode.SUBTRACT_NEXT_FRAME,
        loading_mode: LoadingMode = LoadingMode.RGB,
        dump_memory_snapshot: bool = False,
        flat_conv: bool = False,
        unet_activation: str | None = None,
        attention_reduction: REDUCE_TYPES = "sum",
        attention_only: bool = False,
    ):
        """Initialise the Attention mechanism-based U-Net.

        Args:
            batch_size: Mini-batch size.
            metric: Metric to use for evaluation.
            loss: Loss function to use for training.
            model_type: Architecture to use for the model.
            encoder_name: Name of the encoder.
            encoder_depth: Depth of the encoder.
            encoder_weights: Weights to use for the encoder.
            in_channels: Number of input channels.
            classes: Number of classes.
            num_frames: Number of frames to use.
            weights_from_ckpt_path: Path to checkpoint file.
            optimizer: Optimizer to use.
            optimizer_kwargs: Optimizer keyword arguments.
            scheduler: Learning rate scheduler to use.
            scheduler_kwargs: Scheduler keyword arguments.
            multiplier: Multiplier for the model.
            total_epochs: Total number of epochs.
            alpha: Weight for the loss.
            _beta: (Unused) Weight for the loss.
            learning_rate: Learning rate.
            dl_classification_mode: Classification mode for the dataloader.
            eval_classification_mode: Classification mode for evaluation.
            residual_mode: Residual calculation mode.
            loading_mode: Loading mode for the images.
            dump_memory_snapshot: Whether to dump memory snapshot.
            flat_conv: Whether to use flat convolutions.
            unet_activation: Activation function for the U-Net.
            attention_reduction: Attention reduction type.
            attention_only: Whether to use attention only.

        """
        super().__init__()
        self.save_hyperparameters(ignore=["metric", "loss"])
        self.model_type = model_type
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
        match self.model_type:
            case ModelType.UNET:
                self.model = ResidualAttentionUnet(
                    encoder_name=encoder_name,
                    encoder_depth=encoder_depth,
                    encoder_weights=encoder_weights,
                    residual_mode=residual_mode,
                    in_channels=in_channels,
                    classes=classes,
                    num_frames=num_frames,
                    flat_conv=flat_conv,
                    activation=unet_activation,
                    use_dilations=False,
                    reduce=attention_reduction,
                    _attention_only=attention_only,
                )
            case ModelType.UNET_PLUS_PLUS:
                self.model = ResidualAttentionUnetPlusPlus(
                    encoder_name=encoder_name,
                    encoder_depth=encoder_depth,
                    encoder_weights=encoder_weights,
                    residual_mode=residual_mode,
                    in_channels=in_channels,
                    classes=classes,
                    num_frames=num_frames,
                    flat_conv=flat_conv,
                    activation=unet_activation,
                    use_dilations=True,
                    reduce=attention_reduction,
                    _attention_only=attention_only,
                )
            case _:
                raise NotImplementedError(f"{self.model_type} is not yet implemented!")
        self.residual_mode = residual_mode
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
            self.example_input_array = (
                torch.randn(
                    (self.batch_size, self.num_frames, self.in_channels, 224, 224),
                    dtype=torch.float32,
                ).to(self.device.type),
                torch.randn(
                    (
                        self.batch_size,
                        self.num_frames,
                        (
                            self.in_channels
                            if self.residual_mode == ResidualMode.SUBTRACT_NEXT_FRAME
                            else 2
                        ),
                        224,
                        224,
                    ),
                    dtype=torch.float32,
                ).to(self.device.type),
            )

        self.learning_rate = learning_rate
        self.dl_classification_mode = dl_classification_mode
        self.eval_classification_mode = eval_classification_mode

        # TODO: Move this to setup() method.
        # Sets metric if None.
        self.dice_metrics = {}
        self.other_metrics = {}
        setup_metrics(self, metric, classes)

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

    def forward(self, x_img: torch.Tensor, x_res: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # HACK: This is to get things to work with deepspeed opt level 1 & 2. Level 3
        # is broken due to the casting of batchnorm to non-fp32 types.
        with torch.autocast(device_type=self.device.type):
            return self.model(x_img, x_res)  # pyright: ignore[reportCallIssue]

    @override
    def log_metrics(self, prefix) -> None:
        shared_metric_logging_epoch_end(self, prefix)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
    ):
        """Forward pass for the model with dataloader batches.

        Args:
            batch: Batch of frames, residual frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.

        Return:
            torch.Tensor: Training loss.

        Raises:
            AssertionError: Prediction shape and ground truth mask shapes are different.

        """
        images, res_images, masks, _ = batch
        bs = images.shape[0] if len(images.shape) > 3 else 1
        images_input = images.to(self.device.type)
        res_input = res_images.to(self.device.type)
        masks = masks.to(self.device.type).long()

        with torch.autocast(device_type=self.device.type):
            masks_proba: torch.Tensor = self.model(
                images_input, res_input
            )  # pyright: ignore[reportCallIssue] # False positive

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
            loss_all.item(),
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
            self.train()

        return loss_all

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
    ):
        """Forward pass for the model for one minibatch of a validation epoch.

        Args:
            batch: Batch of frames, residual frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.

        """
        self._shared_eval(batch, batch_idx, "val")

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
    ):
        """Forward pass for the model for one minibatch of a test epoch.

        Args:
            batch: Batch of frames, residual frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.

        """
        self._shared_eval(batch, batch_idx, "test")

    @torch.no_grad()
    def _shared_eval(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
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
        images, res_images, masks, _ = batch
        bs = images.shape[0] if len(images.shape) > 3 else 1
        images_input = images.to(self.device.type)
        res_input = res_images.to(self.device.type)
        masks = masks.to(self.device.type).long()

        masks_proba: torch.Tensor = self.model(
            images_input, res_input
        )  # pyright: ignore[reportCallIssue] # False positive

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

        Return:
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

    @torch.no_grad()
    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str | list[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Forward pass for the model for one minibatch of a test epoch.

        Args:
            batch: Batch of frames, residual frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.
            dataloader_idx: Index of the dataloader.

        Return:
            tuple[torch.Tensor, torch.Tensor, str]: Mask predictions, original images,
                and filename.

        """
        self.eval()
        images, res_images, masks, fn = batch
        images_input = images.to(self.device.type)
        res_input = res_images.to(self.device.type)
        masks = masks.to(self.device.type).long()

        masks_proba: torch.Tensor = self.model(
            images_input, res_input
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
