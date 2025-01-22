"""Implement AFB-URR wrappers for PyTorch Lightning."""

from __future__ import annotations

# Standard Library
from typing import Any, Literal, OrderedDict, override

# Third-Party
from segmentation_models_pytorch.losses import FocalLoss

# PyTorch
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Metric, MetricCollection
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose
from torchvision.utils import draw_segmentation_masks

# State-of-the-Art (SOTA) code
from thirdparty.AFB_URR.model import AFB_URR as AFB_URR_Base
from thirdparty.AFB_URR.model import FeatureBank as FeatureBankBase

# First party imports
from metrics.dice import GeneralizedDiceScoreVariant
from metrics.logging import (
    setup_metrics,
    shared_metric_calculation,
    shared_metric_logging_epoch_end,
)
from utils import utils
from utils.types import (
    INV_NORM_GREYSCALE_DEFAULT,
    INV_NORM_RGB_DEFAULT,
    ClassificationMode,
    LoadingMode,
    MetricMode,
)

# Local folders
from ...common import CommonModelMixin
from .model import AFB_URR, FeatureBank


class AFB_URRLightningModule(CommonModelMixin):
    """AFB-URR LightningModule wrapper."""

    def __init__(
        self,
        batch_size: int,
        metric: Metric | None = None,
        loss: nn.Module | str | None = None,
        encoder_name: str = "resnet50",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        num_frames: int = 30,
        budget: int = 300000,
        weights_from_ckpt_path: str | None = None,
        optimizer: Optimizer | str | None = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: LRScheduler | str | None = "steplr",
        scheduler_kwargs: dict[str, Any] | None = None,
        multiplier: int = 2,
        total_epochs: int = 50,
        alpha: float = 1.0,
        beta: float = 0.5,
        learning_rate: float = 1e-4,
        dl_classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        eval_classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
        dump_memory_snapshot: bool = False,
        use_original: bool = False,
        metric_mode: MetricMode = MetricMode.INCLUDE_EMPTY_CLASS,
        metric_div_zero: float = 1.0,
    ):
        """Initialise the AFB-URR LightningModule wrapper.

        Args:
            batch_size: The batch size.
            metric: The metric to use for evaluation.
            loss: The loss function to use.
            encoder_name: The encoder name.
            encoder_weights: The encoder weights.
            in_channels: The number of input channels.
            classes: The number of classes.
            num_frames: The number of frames.
            budget: The budget for the feature bank.
            weights_from_ckpt_path: The path to the checkpoint.
            optimizer: The optimizer to use.
            optimizer_kwargs: The optimizer keyword arguments.
            scheduler: The scheduler to use.
            scheduler_kwargs: The scheduler keyword arguments.
            multiplier: The multiplier for the model.
            total_epochs: The total number of epochs.
            alpha: The alpha value (scalar for regular loss).
            beta: The beta value (regularisation factor).
            learning_rate: The learning rate.
            dl_classification_mode: The classification mode for the dataloader.
            eval_classification_mode: The classification mode for evaluation.
            loading_mode: The loading mode.
            dump_memory_snapshot: Whether to dump the memory snapshot.
            use_original: Whether to use the original model.
            metric_mode: Metric calculation mode.
            metric_div_zero: How to handle division by zero operations.

        """
        super().__init__()
        self.save_hyperparameters(ignore=["metric", "loss"])
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.classes = classes
        self.num_frames = num_frames
        self.dump_memory_snapshot = dump_memory_snapshot
        self.budget = budget
        self._use_original = use_original
        self.loading_mode = loading_mode
        self.classes = classes

        if self._use_original:
            assert batch_size == 1, "Batch size must be 1 for the original model."

        # Trace memory usage.
        if self.dump_memory_snapshot:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python"
            )

        self.model: (  # pyright: ignore[reportIncompatibleVariableOverride]
            AFB_URR | AFB_URR_Base
        ) = (
            AFB_URR(self.device.type, update_bank=False, load_imagenet_params=True)
            if not self._use_original
            else AFB_URR_Base(self.device, update_bank=False, load_imagenet_params=True)
        )

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}

        if isinstance(loss, str):
            match loss:
                case "cross_entropy":
                    class_weights = torch.Tensor(
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
                case _:
                    raise NotImplementedError(
                        f"Loss type of {loss} is not implemented!"
                    )
        else:
            self.loss = (
                loss
                if isinstance(loss, nn.Module)
                # If none
                else nn.CrossEntropyLoss()
            )

        self.multiplier = multiplier
        self.total_epochs = total_epochs
        self.alpha = alpha
        self.beta = beta
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
        with torch.random.fork_rng(devices=("cpu", self.device)):
            self.example_input_array = (
                torch.randn(
                    (self.batch_size, self.num_frames, self.in_channels, 224, 224),
                    dtype=torch.float32,
                ).to(self.device.type),
                torch.randint(
                    0,
                    1,
                    (self.batch_size, self.classes, 224, 224),
                    dtype=torch.long,
                ).to(self.device.type),
            )

        self.learning_rate = learning_rate
        self.dl_classification_mode = dl_classification_mode
        self.eval_classification_mode = eval_classification_mode

        # TODO: Move this to setup() method.
        # Sets metric if None.
        self.dice_metrics = {}
        self.other_metrics = {}
        setup_metrics(self, metric, classes, metric_mode, metric_div_zero)

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

        self.first_frame_transform = Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomRotation([-90, 90]),
            ]
        )

    def forward(self, frames: Tensor, masks: Tensor) -> Tensor:
        """Forward pass of the model."""
        # HACK: This is to get things to work with deepspeed opt level 1 & 2. Level 3
        # is broken due to the casting of batchnorm to non-fp32 types.
        with torch.autocast(device_type=self.device.type):
            if self._use_original:
                assert isinstance(
                    self.model, AFB_URR_Base
                ), "Wrong model class if use_original is True"

                first_frame = tv_tensors.Image(frames[:, 0].clone())
                first_mask = tv_tensors.Mask(masks.clone())

                first_frame, first_mask = self.first_frame_transform(
                    first_frame, first_mask
                )

                k4_list, v4_list = self.model.memorize(first_frame, first_mask)
                fb_global = FeatureBankBase(self.classes, self.budget, self.device.type)

                fb_global.init_bank(k4_list, v4_list)
                model_ret = self.model.segment(frames[0, 1:], fb_global)
                masks_proba, _uncertainty = model_ret
                masks_proba = masks_proba[-1].unsqueeze(0)
            else:
                assert isinstance(
                    self.model, AFB_URR
                ), "Wrong model class if use_original is False"
                first_frame = tv_tensors.Image(frames[:, 0:1].clone())
                first_mask = tv_tensors.Mask(masks.clone())

                first_frame, first_mask = self.first_frame_transform(
                    first_frame, first_mask
                )
                fb_global = FeatureBank(self.classes, self.budget, self.device.type)

                k4_list, v4_list = self.model.memorize(first_frame, first_mask)

                fb_global.init_bank(k4_list, v4_list)
                model_ret = self.model.segment(frames[:, 1:], fb_global)
                masks_proba, _uncertainty = model_ret
            return masks_proba

    @override
    def log_metrics(self, prefix) -> None:
        shared_metric_logging_epoch_end(self, prefix)

    def _shared_forward_pass(
        self, batch: tuple[Tensor, Tensor, str], prefix: Literal["train", "val", "test"]
    ):
        images, masks, _ = batch
        images_input = images.to(self.device.type)
        masks = masks.to(self.device.type).long()

        if self.dl_classification_mode == ClassificationMode.MULTICLASS_MODE:
            masks_input = F.one_hot(masks, self.classes)
            masks_input = (
                masks_input.permute(0, -1, 1, 2)
                if masks_input.ndim == 4
                else masks_input.permute(-1, 0, 1)
            )
        else:
            masks_input = masks

        with torch.autocast(device_type=self.device.type, enabled=prefix == "train"):
            model_ret: tuple[Tensor, Tensor | None]
            if self._use_original:
                assert isinstance(
                    self.model, AFB_URR_Base
                ), "Wrong model class if use_original is True"

                first_frame = tv_tensors.Image(images_input[:, 0].clone())
                first_mask = tv_tensors.Mask(masks_input.clone())

                first_frame, first_mask = self.first_frame_transform(
                    first_frame, first_mask
                )

                k4_list, v4_list = self.model.memorize(first_frame, first_mask)
                fb_global = FeatureBankBase(self.classes, self.budget, self.device.type)

                fb_global.init_bank(k4_list, v4_list)
                model_ret = self.model.segment(images_input[0, 1:], fb_global)
                masks_proba, uncertainty = model_ret
                masks_proba = masks_proba[-1].unsqueeze(0)
            else:
                assert isinstance(
                    self.model, AFB_URR
                ), "Wrong model class if use_original is False"
                first_frame = tv_tensors.Image(images_input[:, 0:1].clone())
                first_mask = tv_tensors.Mask(masks_input.clone())

                first_frame, first_mask = self.first_frame_transform(
                    first_frame, first_mask
                )
                fb_global = FeatureBank(self.classes, self.budget, self.device.type)

                k4_list, v4_list = self.model.memorize(first_frame, first_mask)

                fb_global.init_bank(k4_list, v4_list)
                model_ret = self.model.segment(images_input[:, 1:], fb_global)
                masks_proba, uncertainty = model_ret

            uncertainty = uncertainty if uncertainty else 0.0

            if (
                self.dl_classification_mode == ClassificationMode.MULTILABEL_MODE
                or self.dl_classification_mode == ClassificationMode.BINARY_CLASS_3_MODE
            ):
                # GUARD: Check that the sizes match.
                assert (
                    masks_proba.size() == masks.size()
                ), f"Output of shape {masks_proba.shape} != target shape: {masks.shape}"
                masks = masks.float()

            # HACK: This ensures that the dimensions to the loss function are correct.
            if isinstance(self.loss, (nn.CrossEntropyLoss, FocalLoss)):
                loss_seg = self.alpha * self.loss(masks_proba, masks.squeeze(dim=1))
            else:
                loss_seg = self.alpha * self.loss(masks_proba, masks)

            loss_all = loss_seg + self.beta * uncertainty

        return loss_all, masks_proba

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, str],
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
        images, masks, _ = batch
        bs = images.shape[0] if len(images.shape) > 3 else 1
        loss_all, masks_proba = self._shared_forward_pass(batch, "train")

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
        batch: tuple[torch.Tensor, torch.Tensor, str],
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
        batch: tuple[torch.Tensor, torch.Tensor, str],
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
        batch: tuple[torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        prefix: Literal["val", "test"],
    ):
        images, masks, _ = batch
        bs = images.shape[0] if len(images.shape) > 3 else 1
        loss_all, masks_proba = self._shared_forward_pass(batch, prefix)

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
                    inv_norm_img[:, -1, :, :, :].detach().cpu(),
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
                    inv_norm_img[:, -1, :, :, :].detach().cpu(),
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
        batch: tuple[torch.Tensor, torch.Tensor, str | list[str]],
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
        images, masks, fn = batch
        images_input = images.to(self.device.type)
        masks = masks.to(self.device.type).long()

        model_ret: tuple[Tensor, Tensor | None]

        if self.dl_classification_mode == ClassificationMode.MULTICLASS_MODE:
            masks_input = F.one_hot(masks, self.classes)
            masks_input = (
                masks_input.permute(0, -1, 1, 2)
                if masks_input.ndim == 4
                else masks_input.permute(-1, 0, 1)
            )
        else:
            masks_input = masks

        if self._use_original:
            assert isinstance(
                self.model, AFB_URR_Base
            ), "Wrong model class if use_original is True"

            first_frame = tv_tensors.Image(images_input[:, 0].clone())
            first_mask = tv_tensors.Mask(masks_input.clone())

            first_frame, first_mask = self.first_frame_transform(
                first_frame, first_mask
            )

            k4_list, v4_list = self.model.memorize(first_frame, first_mask)
            fb_global = FeatureBankBase(self.classes, self.budget, self.device.type)

            fb_global.init_bank(k4_list, v4_list)
            model_ret = self.model.segment(images_input[0, 1:], fb_global)
            masks_proba, uncertainty = model_ret
            masks_proba = masks_proba[-1].unsqueeze(0)
            uncertainty = uncertainty if uncertainty else 0.0
        else:
            assert isinstance(
                self.model, AFB_URR
            ), "Wrong model class if use_original is False"
            first_frame = tv_tensors.Image(images_input[:, 0:1].clone())
            first_mask = tv_tensors.Mask(masks_input.clone())

            first_frame, first_mask = self.first_frame_transform(
                first_frame, first_mask
            )
            fb_global = FeatureBank(self.classes, self.budget, self.device.type)

            k4_list, v4_list = self.model.memorize(first_frame, first_mask)

            fb_global.init_bank(k4_list, v4_list)
            model_ret = self.model.segment(images_input[:, 1:], fb_global)
            masks_proba, uncertainty = model_ret
            uncertainty = uncertainty if uncertainty else 0.0

        if self.eval_classification_mode == ClassificationMode.MULTICLASS_MODE:
            masks_preds = masks_proba.argmax(dim=1)
            masks_preds = F.one_hot(masks_preds, num_classes=4).permute(0, -1, 1, 2)
        else:
            masks_preds = masks_proba > 0.5

        return masks_preds.detach().cpu(), images.detach().cpu(), fn

    @override
    def configure_optimizers(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return utils.configure_optimizers(self)
