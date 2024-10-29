"""Compute mJaccard-Index."""

from typing import Any, Literal, Optional, override

import torch
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassJaccardIndex,
    MultilabelConfusionMatrix,
    MultilabelJaccardIndex,
)
from torchmetrics.functional.classification.confusion_matrix import (
    _multiclass_confusion_matrix_arg_validation,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_update,
    _multilabel_confusion_matrix_arg_validation,
    _multilabel_confusion_matrix_format,
    _multilabel_confusion_matrix_tensor_validation,
    _multilabel_confusion_matrix_update,
)
from torchmetrics.functional.classification.jaccard import (
    _jaccard_index_reduce,
    _multiclass_jaccard_index_arg_validation,
    _multilabel_jaccard_index_arg_validation,
)


class MulticlassMJaccardIndex(MulticlassJaccardIndex):
    """Calculate the mJaccard Index for multiclass tasks."""

    mJaccard_running: torch.Tensor
    samples: torch.Tensor

    @override
    def __init__(
        self,
        num_classes: int,
        average: Optional[Literal["macro", "none"]],
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        zero_division: float = 0,
        **kwargs: Any,
    ) -> None:
        super(MulticlassConfusionMatrix, self).__init__(**kwargs)
        if validate_args:
            _multiclass_confusion_matrix_arg_validation(
                num_classes, ignore_index, normalize=None
            )
            _multiclass_jaccard_index_arg_validation(num_classes, ignore_index, average)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.average: Optional[Literal["macro", "none"]] = average
        self.zero_division = zero_division
        self.normalize = None
        self.add_state(
            "mJaccard_running",
            torch.zeros(num_classes, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "samples", torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum"
        )

    @override
    def compute(self):
        match self.average:
            case "macro":
                return (self.mJaccard_running / self.samples).mean()
            case _:
                return self.mJaccard_running / self.samples

    @override
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        bs = preds.shape[0]
        self.samples += bs

        if self.validate_args:
            _multiclass_confusion_matrix_tensor_validation(
                preds, target, self.num_classes, self.ignore_index
            )
        preds, target = _multiclass_confusion_matrix_format(
            preds, target, self.ignore_index
        )
        confmat = _multiclass_confusion_matrix_update(preds, target, self.num_classes)
        jaccard = _jaccard_index_reduce(
            confmat, average=None, zero_division=self.zero_division
        )

        self.mJaccard_running += jaccard * bs


class MultilabelMJaccardIndex(MultilabelJaccardIndex):
    """Calculate the mJaccard Index for multilabel tasks."""

    mJaccard_running: torch.Tensor
    samples: torch.Tensor

    @override
    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: Optional[Literal["macro", "none"]] = "macro",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super(MultilabelConfusionMatrix, self).__init__(**kwargs)
        if validate_args:
            _multilabel_confusion_matrix_arg_validation(
                num_labels, threshold, ignore_index, None
            )
            _multilabel_jaccard_index_arg_validation(
                num_labels, threshold, ignore_index, average
            )
        self.num_labels = num_labels
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.normalize = None
        self.validate_args = validate_args
        self.average: Optional[Literal["macro", "none"]] = average

        self.add_state(
            "mJaccard_running",
            torch.zeros(num_labels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "samples", torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum"
        )

    @override
    def compute(self):
        match self.average:
            case "macro":
                return (self.mJaccard_running / self.samples).mean()
            case _:
                return self.mJaccard_running / self.samples

    @override
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        bs = preds.shape[0]
        self.samples += bs

        if self.validate_args:
            _multilabel_confusion_matrix_tensor_validation(
                preds, target, self.num_labels, self.ignore_index
            )

        preds, target = _multilabel_confusion_matrix_format(
            preds, target, self.num_labels, self.threshold, self.ignore_index
        )
        confmat = _multilabel_confusion_matrix_update(preds, target, self.num_labels)
        jaccard = _jaccard_index_reduce(
            confmat, average=None, zero_division=self.zero_division
        )

        self.mJaccard_running += jaccard * bs
