"""Common definitions for the models module."""

from typing import Literal, Union, override

import lightning as L
import torch
from huggingface_hub import ModelHubMixin
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from segmentation_models_pytorch.base.model import SegmentationModel
from torch import nn
from torchmetrics import Metric, MetricCollection
from torchvision.transforms.v2 import Compose

from utils.types import ClassificationMode, InverseNormalize, ModelType


class CommonModelMixin(L.LightningModule):
    """Common model attributes."""

    dl_classification_mode: ClassificationMode
    eval_classification_mode: ClassificationMode
    dice_metrics: dict[str, MetricCollection | Metric]
    other_metrics: dict[str, MetricCollection]
    model: Union[SegmentationModel, ModelHubMixin, nn.Module]
    model_type: ModelType
    de_transform: Compose | InverseNormalize

    @override
    def on_train_start(self):
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.log_hyperparams(
                self.hparams,  # pyright: ignore[reportArgumentType]
                {
                    # (1) Validation Loss
                    "hp/val_loss": 0,
                    # (2) Dice score
                    "hp/val/dice_macro_avg": 0,
                    "hp/val/dice_weighted_avg": 0,
                    "hp/val/dice_macro_class_2_3": 0,
                    "hp/val/dice_weighted_class_2_3": 0,
                    "hp/val/dice_class_1": 0,
                    "hp/val/dice_class_2": 0,
                    "hp/val/dice_class_3": 0,
                    # (3) Jaccard Index
                    "hp/val/jaccard_macro_avg": 0,
                    "hp/val/jaccard_class_1": 0,
                    "hp/val/jaccard_class_2": 0,
                    "hp/val/jaccard_class_3": 0,
                    # (4) Precision
                    "hp/val/precision_macro_avg": 0,
                    "hp/val/precision_class_1": 0,
                    "hp/val/precision_class_2": 0,
                    "hp/val/precision_class_3": 0,
                    # (5) Recall
                    "hp/val/recall_macro_avg": 0,
                    "hp/val/recall_class_1": 0,
                    "hp/val/recall_class_2": 0,
                    "hp/val/recall_class_3": 0,
                },
            )

    def log_metrics(self, prefix: Literal["train", "val", "test"]) -> None:
        """Implement shared metric logging epoch end here.

        Note: This is to prevent circular imports with the logging module.
        """
        raise NotImplementedError("Log metrics not implemented!")

    @override
    def on_train_end(self) -> None:
        if self.dump_memory_snapshot:
            torch.cuda.memory._dump_snapshot("two_plus_one_snapshot.pickle")

    @override
    def on_train_epoch_end(self) -> None:
        self.log_metrics("train")

    @override
    def on_validation_epoch_end(self) -> None:
        self.log_metrics("val")

    @override
    def on_test_epoch_end(self) -> None:
        self.log_metrics("test")


ENCODER_OUTPUT_SHAPES = {
    "resnet18": [
        (64, 112, 112),
        (64, 56, 56),
        (128, 28, 28),
        (256, 14, 14),
        (512, 7, 7),
    ],
    "resnet34": [
        (64, 112, 112),
        (64, 56, 56),
        (128, 28, 28),
        (256, 14, 14),
        (512, 7, 7),
    ],
    "resnet50": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "resnet101": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "resnet152": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "resnext50_32x4d": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "resnext101_32x4d": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "resnext101_32x8d": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "resnext101_32x16d": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "resnext101_32x32d": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "resnext101_32x48d": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "senet154": [
        (128, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "se_resnet50": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "se_resnet101": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "se_resnet152": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "se_resnext50_32x4d": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "se_resnext101_32x4d": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "tscsenet154": [
        (128, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "tscse_resnet50": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "tscse_resnet101": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
    "tscse_resnet152": [
        (64, 112, 112),
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ],
}
"""Output shapes for the different ResNet models. The output shapes are used to
calculate the number of output channels for each 1D temporal convolutional block.
"""
