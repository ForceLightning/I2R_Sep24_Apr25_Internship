"""Implement Vivim wrappers for PyTorch Lightning."""

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
from torchvision.transforms.v2 import Compose
from torchvision.utils import draw_segmentation_masks

# State-of-the-Art (SOTA) code
from thirdparty.vivim.modeling.vivim import Vivim

# First party imports
from metrics.dice import GeneralizedDiceScoreVariant
from metrics.logging import (
    setup_metrics,
    shared_metric_calculation,
    shared_metric_logging_epoch_end,
)
from metrics.loss import JointEdgeSegLoss, StructureLoss
from models.common import CommonModelMixin
from utils import utils
from utils.types import (
    INV_NORM_GREYSCALE_DEFAULT,
    INV_NORM_RGB_DEFAULT,
    ClassificationMode,
    LoadingMode,
    MetricMode,
)


class VivimLightningModule(CommonModelMixin):
    """Vivim LightningModule wrapper."""

    _default_depths = [2, 2, 2, 2]
    _default_feat_size = [64, 128, 320, 512]
    _default_optimizer = "adam"
    _default_optimizer_kwargs = {"betas": [0.9, 0.999]}
    _default_scheduler = "cosine_anneal"

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
        depths: list[int] = _default_depths,
        feat_size: list[int] = _default_feat_size,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        hidden_size: int = 768,
        norm_name: str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims: int = 2,
        with_edge: bool = False,
        weights_from_ckpt_path: str | None = None,
        optimizer: Optimizer | str | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: LRScheduler | str | None = None,
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
        metric_mode: MetricMode = MetricMode.INCLUDE_EMPTY_CLASS,
        metric_div_zero: float = 1.0,
    ):
        """Initialise the Vivim LightningModule.

        Args:
            batch_size: The batch size.
            metric: The metric to use.
            loss: The loss function to use.
            encoder_name: The name of the encoder.
            encoder_depth: The depth of the encoder.
            encoder_weights: The weights of the encoder.
            in_channels: The number of input channels.
            classes: The number of classes.
            num_frames: The number of frames.
            depths: The number of modules per layer.
            feat_size: The feature sizes.
            drop_path_rate: The dropout rate.
            layer_scale_init_value: The layer scale initial value.
            hidden_size: The hidden size.
            norm_name: The name of the normalisation.
            conv_block: Whether to use convolutional blocks.
            res_block: Whether to use residual blocks.
            spatial_dims: The spatial dimensions.
            with_edge: Whether to use edges.
            weights_from_ckpt_path: The path to the checkpoint.
            optimizer: The optimizer to use.
            optimizer_kwargs: The optimizer keyword arguments.
            scheduler: The scheduler to use.
            scheduler_kwargs: The scheduler keyword arguments.
            multiplier: The multiplier.
            total_epochs: The total number of epochs.
            alpha: The alpha loss scaling value.
            _beta: The beta loss scaling value. (Unused).
            learning_rate: The learning rate.
            dl_classification_mode: The classification mode for dataloader.
            eval_classification_mode: The classification mode for evaluation.
            loading_mode: The image loading mode.
            dump_memory_snapshot: Whether to dump memory snapshot.
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
        self.classes = classes

        # Trace memory usage.
        if self.dump_memory_snapshot:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python"
            )

        self.model: Vivim = (  # pyright: ignore[reportIncompatibleVariableOverride]
            Vivim(
                in_chans=in_channels,
                out_chans=classes,
                depths=depths,
                feat_size=feat_size,
                drop_path_rate=drop_path_rate,  # pyright: ignore[reportArgumentType]
                layer_scale_init_value=layer_scale_init_value,
                hidden_size=hidden_size,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
                spatial_dims=spatial_dims,
                with_edge=with_edge,
            )
        )

        if optimizer is None and optimizer_kwargs is None:
            self.optimizer = self._default_optimizer
            self.optimizer_kwargs = self._default_optimizer_kwargs | {
                "lr": learning_rate
            }
        else:
            self.optimizer = optimizer if optimizer else "adamw"
            self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}

        if scheduler is None and scheduler_kwargs is None:
            self.scheduler = self._default_scheduler
            self.scheduler_kwargs = {
                "T_max": total_epochs,
                "eta_min": learning_rate * 0.01,
            }
        else:
            self.scheduler = scheduler if scheduler else "gradual_warmup_scheduler"
            self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}

        self.loading_mode = loading_mode

        if isinstance(loss, str):
            match loss:
                case "cross_entropy":
                    self.loss = StructureLoss()
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
                else (
                    JointEdgeSegLoss(num_classes=classes)
                    if with_edge
                    else StructureLoss()
                )
            )
        self.with_edge = with_edge

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
        # Input shape is (B, F, C, H, W)
        with torch.random.fork_rng(devices=("cpu", "cuda:0")):
            self.example_input_array = torch.randn(
                (self.batch_size, self.num_frames, self.in_channels, 224, 224),
                dtype=torch.float32,
            ).to(self.device.type)

        self.learning_rate = learning_rate
        self.dl_classification_mode = dl_classification_mode
        self.eval_classification_mode = eval_classification_mode

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

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor] | Tensor:
        with torch.autocast(device_type=self.device.type):
            return self.model(x)  # pyright: ignore[reportCallIssue]

    @override
    def training_step(
        self,
        batch: tuple[Tensor, Tensor, str] | tuple[Tensor, Tensor, Tensor, str],
        batch_idx: int,
    ) -> Tensor:
        if len(batch) == 3:
            images, masks, _ = batch
            edge_gt = None
        else:
            images, masks, edge_gt, _ = batch
        bs = images.shape[0] if images.ndim > 3 else 1
        h, w = images.shape[-2:]
        images_input = images.to(self.device.type, dtype=torch.float32)
        masks = masks.to(self.device.type).long()

        with torch.autocast(device_type=self.device.type):
            if not self.with_edge:
                masks_proba = self.model(
                    images_input
                )  # pyright: ignore[reportCallIssue]
                e0 = None
            else:
                masks_proba, e0 = self.model(
                    images_input
                )  # pyright: ignore[reportCallIssue]
                e0 = e0.reshape(bs, self.num_frames, 1, h, w)
                e0 = e0[:, 0, :, :, :]

        masks_proba = masks_proba.reshape(bs, self.num_frames, self.classes, h, w)
        masks_proba = masks_proba[:, 0, :, :, :]

        if (
            self.dl_classification_mode == ClassificationMode.MULTILABEL_MODE
            or self.dl_classification_mode == ClassificationMode.BINARY_CLASS_3_MODE
        ):
            # GUARD: Check that the sizes match.
            assert (
                masks_proba.size() == masks.size()
            ), f"Output of shape {masks_proba.shape} != target shape: {masks.shape}"

        if not self.with_edge:
            # HACK: This ensures that the dimensions to the loss function are correct.
            if isinstance(self.loss, (nn.CrossEntropyLoss, FocalLoss, StructureLoss)):
                loss_seg = self.alpha * self.loss(masks_proba, masks.squeeze(dim=1))
            else:
                loss_seg = self.alpha * self.loss(masks_proba, masks)
        else:
            assert e0 is not None, "edge output is None!"
            assert edge_gt is not None, "edge ground truth is None!"
            assert isinstance(
                self.loss, JointEdgeSegLoss
            ), f"self.loss is not of type JointEdgeSegLoss: {self.loss.__class__}"

            loss_seg = self.loss((masks_proba, e0), (masks, edge_gt))

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

    @override
    def validation_step(
        self,
        batch: tuple[Tensor, Tensor, str],
        batch_idx: int,
    ):
        self._shared_eval(batch, batch_idx, "val")

    @override
    def test_step(
        self,
        batch: tuple[Tensor, Tensor, str],
        batch_idx: int,
    ):
        self._shared_eval(batch, batch_idx, "test")

    @torch.no_grad()
    def _shared_eval(
        self,
        batch: tuple[Tensor, Tensor, str] | tuple[Tensor, Tensor, Tensor, str],
        batch_idx: int,
        prefix: Literal["val", "test"],
    ):
        if len(batch) == 3:
            images, masks, _ = batch
            edge_gt = None
        else:
            images, masks, edge_gt, _ = batch

        bs = images.shape[0] if images.ndim > 3 else 1
        images_input = images.to(self.device.type, dtype=torch.float32)
        h, w = images.shape[-2:]
        masks = masks.to(self.device.type).long()

        if not self.with_edge:
            masks_proba = self.model(images_input)  # pyright: ignore[reportCallIssue]
            e0 = None
        else:
            masks_proba, e0 = self.model(
                images_input
            )  # pyright: ignore[reportCallIssue]
            e0 = e0.reshape(bs, self.num_frames, 1, h, w)
            e0 = e0[:, 0, :, :, :]

        masks_proba = masks_proba.reshape(bs, self.num_frames, self.classes, h, w)
        masks_proba = masks_proba[:, 0, :, :, :]

        if (
            self.dl_classification_mode == ClassificationMode.MULTILABEL_MODE
            or self.dl_classification_mode == ClassificationMode.BINARY_CLASS_3_MODE
        ):
            # GUARD: Check that the sizes match.
            assert (
                masks_proba.size() == masks.size()
            ), f"Output of shape {masks_proba.shape} != target shape: {masks.shape}"

        if not self.with_edge:
            # HACK: This ensures that the dimensions to the loss function are correct.
            if isinstance(self.loss, (nn.CrossEntropyLoss, FocalLoss, StructureLoss)):
                loss_seg = self.alpha * self.loss(masks_proba, masks.squeeze(dim=1))
            else:
                loss_seg = self.alpha * self.loss(masks_proba, masks)
        else:
            assert e0 is not None, "edge output is None!"
            assert edge_gt is not None, "edge ground truth is None!"
            assert isinstance(
                self.loss, JointEdgeSegLoss
            ), f"self.loss is not of type JointEdgeSegLoss: {self.loss.__class__}"

            loss_seg = self.loss((masks_proba, e0), (masks, edge_gt))

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

    @override
    def log_metrics(self, prefix: Literal["train", "val", "test"]) -> None:
        shared_metric_logging_epoch_end(self, prefix)

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
        batch: (
            tuple[torch.Tensor, torch.Tensor, str | list[str]]
            | tuple[Tensor, Tensor, Tensor, str | list[str]]
        ),
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[Tensor, Tensor, str | list[str]]:
        """Forward pass for the model for one minibatch of a test epoch.

        Args:
            batch: Batch of frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.
            dataloader_idx: Index of the dataloader.

        Returns:
            Mask predictions, original images, and filename.

        """
        self.eval()
        if len(batch) == 3:
            images, masks, fn = batch
        else:
            images, masks, _, fn = batch

        bs = images.shape[0] if images.ndim > 3 else 1
        h, w = images.shape[-2:]

        images_input = images.to(self.device.type)
        masks = masks.to(self.device.type).long()

        masks_proba: Tensor
        if self.with_edge:
            masks_proba, _ = self.model(
                images_input
            )  # pyright: ignore[reportCallIssue]
        else:
            masks_proba = self.model(images_input)  # pyright: ignore[reportCallIssue]

        masks_proba = masks_proba.reshape(bs, self.num_frames, self.classes, h, w)
        masks_proba = masks_proba[:, 0, :, :, :]

        if self.eval_classification_mode == ClassificationMode.MULTICLASS_MODE:
            masks_preds = masks_proba.argmax(dim=1)
            masks_preds = F.one_hot(masks_preds, num_classes=4).permute(0, -1, 1, 2)
        else:
            masks_preds = masks_proba > 0.5

        return masks_preds.detach().cpu(), images.detach().cpu(), fn

    @override
    def configure_optimizers(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return utils.configure_optimizers(self)
