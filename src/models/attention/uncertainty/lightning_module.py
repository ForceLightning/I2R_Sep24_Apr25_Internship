# -*- coding: utf-8 -*-
"""Uncertainty LightningModule wrappers for U-Net with attention mechanism."""

from __future__ import annotations

# Standard Library
from typing import Any, Literal, OrderedDict, override

# Third-Party
import segmentation_models_pytorch as smp
from einops import rearrange
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

# PyTorch
import torch
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch_uncertainty.metrics import (
    AUGRC,
    AURC,
    BrierScore,
    CalibrationError,
    CategoricalNLL,
    MeanIntersectionOverUnion,
)
from torch_uncertainty.models import EPOCH_UPDATE_MODEL, STEP_UPDATE_MODEL
from torch_uncertainty.routines.segmentation import (
    SegmentationRoutine,
    _segmentation_routine_checks,
)
from torchmetrics import Accuracy, Metric, MetricCollection
from torchvision.transforms.v2 import Compose

# First party imports
from metrics.dice import GeneralizedDiceScoreVariant
from metrics.logging import (
    setup_metrics,
    shared_metric_calculation,
    shared_metric_logging_epoch_end,
)
from models.two_plus_one import TemporalConvolutionalType
from utils import utils
from utils.types import (
    INV_NORM_GREYSCALE_DEFAULT,
    INV_NORM_RGB_DEFAULT,
    ClassificationMode,
    LoadingMode,
    ModelType,
    ResidualMode,
)

# Local folders
from ...common import CommonModelMixin
from ..model import REDUCE_TYPES
from ..segmentation_model import ResidualAttentionUnet, ResidualAttentionUnetPlusPlus
from .uncertainty import MCDropout


class UncertaintyResidualAttentionLightningModule(
    CommonModelMixin, SegmentationRoutine
):
    """Attention mechanism-based U-Net with uncertainty."""

    def __init__(
        self,
        batch_size: int,
        classes: int,
        metric: Metric | None = None,
        loss: nn.Module | str | None = None,
        model_type: ModelType = ModelType.UNET,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        num_frames: int = 5,
        weights_from_ckpt_path: str | None = None,
        mc_dropout_estimators: int = 4,
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
        dummy_predict: bool = False,
        eval_shift: bool = False,
        format_batch_fn: nn.Module | None = None,
        metric_subsampling_rate: float = 0.01,
        log_plots: bool = False,
        num_samples_to_plot: int = 3,
        num_calibration_bins: int = 15,
        temporal_conv_type: TemporalConvolutionalType = TemporalConvolutionalType.ORIGINAL,
    ) -> None:
        """Initialise the Attention mechanism-based U-Net.

        Args:
            batch_size: Mini-batch size.
            classes: Number of classes.
            metric: Metric to use for evaluation.
            loss: Loss function to use for training.
            model_type: Architecture to use for the model.
            encoder_name: Name of the encoder.
            encoder_depth: Depth of the encoder.
            encoder_weights: Weights to use for the encoder.
            in_channels: Number of input channels.
            num_frames: Number of frames to use.
            weights_from_ckpt_path: Path to checkpoint file.
            mc_dropout_estimators: Number of iterations to run validation for
                monte-carlo dropout.
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
            dummy_predict: Whether to predict the ground truth for visualisation.
            eval_shift: Indicates whether to evaluate the Distribution shift
                performance. Defaults to ``False``.
            format_batch_fn: The function to format the batch. Defaults to ``None``.
            metric_subsampling_rate: The rate of subsampling for the memory consuming
                metrics. Defaults to ``1e-2``.
            log_plots: Indicates whether to log plots from metrics. Defaults to
                ``False``.
            num_samples_to_plot: Number of samples to plot in the segmentation results.
                Defaults to ``3``.
            num_calibration_bins: Number of bins to compute calibration metrics.
                Defaults to ``15``.
            temporal_conv_type: What kind of temporal convolutional layers to use.

        """
        super(SegmentationRoutine, self).__init__()
        self.save_hyperparameters(ignore=["metric", "loss"])

        _segmentation_routine_checks(
            classes,
            metric_subsampling_rate,
            num_calibration_bins,
        )
        if eval_shift:
            raise NotImplementedError(
                "Distribution shift evaluation not implemented yet. Raise an issue "
                "if needed."
            )

        self.save_hyperparameters(ignore=["metric", "loss"])
        self.model_type = model_type
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.classes = classes
        self.num_classes = classes
        self.num_frames = num_frames
        self.dump_memory_snapshot = dump_memory_snapshot
        self.dummy_predict = dummy_predict
        self.residual_mode = residual_mode
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}
        self.loading_mode = loading_mode
        self.multiplier = multiplier
        self.total_epochs = total_epochs
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.dl_classification_mode = dl_classification_mode
        self.eval_classification_mode = eval_classification_mode
        self.mc_dropout_estimators = mc_dropout_estimators

        if format_batch_fn is None:
            format_batch_fn = nn.Identity()

        self.format_batch_fn = format_batch_fn
        self.metric_subsampling_rate = metric_subsampling_rate
        self.log_plots = log_plots

        # Trace memory usage
        if self.dump_memory_snapshot:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python"
            )
        # PERF: The model can be `torch.compile()`'d but layout issues occur with
        # convolutional networks. See: https://github.com/pytorch/pytorch/issues/126585
        match self.model_type:
            case ModelType.UNET:
                model = ResidualAttentionUnet(  # pyright: ignore[reportAttributeAccessIssue]
                    encoder_name=encoder_name,
                    encoder_depth=encoder_depth,
                    encoder_weights=encoder_weights,
                    residual_mode=residual_mode,
                    in_channels=in_channels,
                    classes=self.classes,
                    num_frames=num_frames,
                    flat_conv=flat_conv,
                    activation=unet_activation,
                    temporal_conv_type=temporal_conv_type,
                    reduce=attention_reduction,
                    _attention_only=attention_only,
                )
            case ModelType.UNET_PLUS_PLUS:
                model = ResidualAttentionUnetPlusPlus(  # pyright: ignore[reportAttributeAccessIssue]
                    encoder_name=encoder_name,
                    encoder_depth=encoder_depth,
                    encoder_weights=encoder_weights,
                    residual_mode=residual_mode,
                    in_channels=in_channels,
                    classes=self.classes,
                    num_frames=num_frames,
                    flat_conv=flat_conv,
                    activation=unet_activation,
                    temporal_conv_type=temporal_conv_type,
                    reduce=attention_reduction,
                    _attention_only=attention_only,
                )
            case _:
                raise NotImplementedError(f"{self.model_type} is not yet implemented!")

        self.model = MCDropout(
            model,  # pyright: ignore[reportArgumentType]
            num_estimators=self.mc_dropout_estimators,
            last_layer=False,
            on_batch=True,
        )

        self.needs_epoch_update = isinstance(model, EPOCH_UPDATE_MODEL)
        self.needs_step_update = isinstance(model, STEP_UPDATE_MODEL)

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

        # TODO: Move this to setup() method.
        # Sets metric if None.
        self.dice_metrics = {}
        self.other_metrics = {}
        setup_metrics(self, metric, self.classes)

        # Uncertainty metrics
        seg_metrics = MetricCollection(
            {
                "seg/mIoU": MeanIntersectionOverUnion(num_classes=classes),
            },
            compute_groups=False,
        )
        sbsmpl_seg_metrics = MetricCollection(
            {
                "seg/mAcc": Accuracy(
                    task="multiclass", average="macro", num_classes=classes
                ),
                "seg/Brier": BrierScore(num_classes=classes),
                "seg/NLL": CategoricalNLL(),
                "seg/pixAcc": Accuracy(task="multiclass", num_classes=classes),
                "cal/ECE": CalibrationError(
                    task="multiclass",
                    num_classes=classes,
                    num_bins=num_calibration_bins,
                ),
                "cal/aECE": CalibrationError(
                    task="multiclass",
                    adaptive=True,
                    num_bins=num_calibration_bins,
                    num_classes=classes,
                ),
                "sc/AURC": AURC(),
                "sc/AUGRC": AUGRC(),
            },
            compute_groups=[
                ["seg/mAcc"],
                ["seg/Brier"],
                ["seg/NLL"],
                ["seg/pixAcc"],
                ["cal/ECE", "cal/aECE"],
                ["sc/AURC", "sc/AUGRC"],
            ],
        )

        self.val_seg_metrics = seg_metrics.clone(prefix="val/")
        self.val_sbsmpl_seg_metrics = sbsmpl_seg_metrics.clone(prefix="val/")
        self.test_seg_metrics = seg_metrics.clone(prefix="test/")
        self.test_sbsmpl_seg_metrics = sbsmpl_seg_metrics.clone(prefix="test/")

        if log_plots:
            self.num_samples_to_plot = num_samples_to_plot
            self.sample_buffer = []

        # Attempts to load checkpoint if provided.
        self.weights_from_ckpt_path = weights_from_ckpt_path
        if self.weights_from_ckpt_path:
            ckpt: dict[str, dict[str, Any]] = torch.load(self.weights_from_ckpt_path)
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
            except RuntimeError as e:
                try:
                    # HACK: Load models without an MC wrapper.
                    new_state_dict = OrderedDict()
                    for k, v in ckpt["state_dict"].items():
                        name = k
                        if "model." in k:
                            name = k.replace("model.", "core_model.")
                        new_state_dict[name] = v
                    self.model.load_state_dict(new_state_dict)
                except Exception as e2:
                    raise e2 from e

    @override
    def configure_optimizers(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return utils.configure_optimizers(self)

    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, x_img: Tensor, x_res: Tensor
    ) -> Tensor:
        with torch.autocast(device_type=self.device.type, enabled=self.training):
            return self.model(x_img, x_res)

    @override
    def on_train_start(self):
        super(CommonModelMixin, self).on_train_start()

    @override
    def training_step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, batch: tuple[Tensor, Tensor, Tensor, str], batch_idx: int
    ) -> STEP_OUTPUT:
        images, res_images, masks, _ = batch
        masks = masks.to(self.device.type).long()
        images_input = images.to(self.device.type)
        res_input = res_images.to(self.device.type)
        bs = images.shape[0] if len(images.shape) > 3 else 1

        masks_proba = self.forward(images_input, res_input)

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

    @torch.no_grad()
    def _shared_eval(
        self,
        batch: tuple[Tensor, Tensor, Tensor, str],
        batch_idx: int,
        prefix: Literal["val", "test"],
    ):
        self.eval()
        images, res_images, masks, _ = batch
        masks = masks.to(self.device.type).long()
        images_input = images.to(self.device.type)
        res_input = res_images.to(self.device.type)
        bs = images.shape[0] if len(images.shape) > 3 else 1

        masks_proba = self.forward(images_input, res_input)  # (B x M x C x H x W)

        # Compute loss and log it.
        # HACK: This ensures that the dimensions to the loss function are correct.
        if isinstance(self.loss, nn.CrossEntropyLoss) or isinstance(
            self.loss, FocalLoss
        ):
            loss_seg = self.alpha * self.loss(masks_proba.mean(1), masks.squeeze(dim=1))
        else:
            loss_seg = self.alpha * self.loss(masks_proba.mean(1), masks)

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

        # Calculate regular metrics and log images.
        if isinstance(
            self.dice_metrics[prefix], GeneralizedDiceScoreVariant
        ) or isinstance(self.dice_metrics[prefix], MetricCollection):
            masks_preds, masks_one_hot = shared_metric_calculation(
                self, masks, masks_proba.mean(dim=1), prefix
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

        # Compute uncertainty metrics
        logits = rearrange(masks_proba, "b m c h w -> (b h w) m c")
        probs_per_est = logits.softmax(dim=-1)
        probs = probs_per_est.mean(dim=1)
        std = probs_per_est.std(dim=1)
        targets = masks.flatten()

        return probs, std, targets

    @override
    def validation_step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, batch: tuple[Tensor, Tensor, Tensor, str], batch_idx: int
    ):
        probs, _, targets = self._shared_eval(batch, batch_idx, "val")

        self.val_seg_metrics.update(probs, targets)
        self.val_sbsmpl_seg_metrics.update(*self.subsample(probs, targets))

    @override
    def test_step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, batch: tuple[Tensor, Tensor, Tensor, str], batch_idx: int
    ):
        probs, _, targets = self._shared_eval(batch, batch_idx, "test")

        self.test_seg_metrics.update(probs, targets)
        self.test_sbsmpl_seg_metrics.update(*self.subsample(probs, targets))

    @override
    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        res_dict = self.val_seg_metrics.compute()
        self.log_dict(res_dict, logger=True, sync_dist=True)
        self.log(
            "mIoU%",
            res_dict["val/seg/mIoU"] * 100,
            prog_bar=True,
            sync_dist=True,
        )
        self.log_dict(self.val_sbsmpl_seg_metrics.compute(), sync_dist=True)
        self.val_seg_metrics.reset()
        self.val_sbsmpl_seg_metrics.reset()

    @override
    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        self.log_dict(self.test_seg_metrics.compute(), sync_dist=True)
        self.log_dict(self.test_sbsmpl_seg_metrics.compute(), sync_dist=True)
        if isinstance(self.logger, TensorBoardLogger) and self.log_plots:
            self.logger.experiment.add_figure(
                "Calibration/Reliabity diagram",
                self.test_sbsmpl_seg_metrics["cal/ECE"].plot()[0],
            )
            self.logger.experiment.add_figure(
                "Selective Classification/Risk-Coverage curve",
                self.test_sbsmpl_seg_metrics["sc/AURC"].plot()[0],
            )
            self.logger.experiment.add_figure(
                "Selective Classification/Generalized Risk-Coverage curve",
                self.test_sbsmpl_seg_metrics["sc/AUGRC"].plot()[0],
            )
            self.log_segmentation_plots()

    @torch.no_grad()
    def predict_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, str | list[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Forward pass for the model for one minibatch of a test epoch.

        Args:
            batch: Batch of frames, residual frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.
            dataloader_idx: Index of the dataloader.

        Return:
            tuple[Tensor, Tensor, str]: Mask predictions, original images,
                and filename.

        """
        self.eval()
        images, res_images, masks, fn = batch
        images_input = images.to(self.device.type)
        res_input = res_images.to(self.device.type)
        masks = masks.to(self.device.type).long()

        masks_preds: Tensor
        assert not self.dummy_predict
        assert isinstance(self.model, MCDropout)

        for mod in self.model.filtered_modules:
            mod.train(False)

        # (B, C, H, W)
        masks_proba: Tensor = self.model.core_model(images_input, res_input)

        if self.eval_classification_mode == ClassificationMode.MULTICLASS_MODE:
            masks_preds = masks_proba.argmax(dim=1)
            masks_preds = (
                F.one_hot(masks_preds, num_classes=4).permute(0, -1, 1, 2).bool()
            )
        else:
            masks_preds = masks_proba.mean(dim=1) > 0.5

        self.eval()
        assert all(fm.training for fm in self.model.filtered_modules)

        # Reference: https://github.com/tha-santacruz/BayesianUNet
        # (B, M, C, H, W)
        dropout_predictions = self.model(images_input, res_input)
        dropout_predictions = rearrange(
            dropout_predictions, "b m k h w -> m b k h w"
        ).softmax(dim=2)
        batch_mean = dropout_predictions.mean(dim=0)

        entropy = -torch.sum(batch_mean * batch_mean.log(), dim=1)
        mutual_info = entropy + torch.mean(
            torch.sum(dropout_predictions * dropout_predictions.log(), dim=-3), dim=0
        )

        # Rescale to [0, 1]
        mutual_info = (mutual_info - mutual_info.min()) / (
            mutual_info.max() - mutual_info.min()
        )

        return (
            masks_preds.detach().cpu(),
            mutual_info.detach().cpu(),
            images.detach().cpu(),
            fn,
        )

    @override
    def log_metrics(self, prefix: Literal["train", "val", "test"]) -> None:
        shared_metric_logging_epoch_end(self, prefix)
