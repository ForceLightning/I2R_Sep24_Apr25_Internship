"""Compute mPrecision and mRecall."""

from typing import Any, Literal, Optional, override

import torch
from torch import Tensor
from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelPrecision,
    MultilabelRecall,
)
from torchmetrics.classification.stat_scores import _AbstractStatScores
from torchmetrics.functional.classification.precision_recall import (
    _precision_recall_reduce,
)
from torchmetrics.functional.classification.stat_scores import (
    _multiclass_stat_scores_arg_validation,
    _multiclass_stat_scores_format,
    _multiclass_stat_scores_tensor_validation,
    _multiclass_stat_scores_update,
    _multilabel_stat_scores_arg_validation,
    _multilabel_stat_scores_format,
    _multilabel_stat_scores_tensor_validation,
    _multilabel_stat_scores_update,
)


class MulticlassMPrecision(MulticlassPrecision):
    """Calculates mPrecision for multiclass tasks."""

    mPrecision_running: Tensor
    samples: Tensor

    @override
    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        average: Optional[Literal["macro", "none"]] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ):
        zero_division = kwargs.pop("zero_division", 0)
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _multiclass_stat_scores_arg_validation(
                num_classes,
                top_k,
                average,
                multidim_average,
                ignore_index,
                zero_division,
            )
        self.num_classes = num_classes
        self.top_k = top_k
        self.average: Optional[Literal["macro", "none"]] = average
        self.multidim_average: Literal["global", "samplewise"] = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        self.add_state(
            "mPrecision_running",
            torch.zeros(num_classes, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "samples", torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum"
        )

    @override
    def compute(self) -> Tensor:
        avg = self.mPrecision_running / self.samples
        return avg

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.validate_args:
            _multiclass_stat_scores_tensor_validation(
                preds,
                target,
                self.num_classes,
                self.multidim_average,
                self.ignore_index,
            )

        preds, target = _multiclass_stat_scores_format(preds, target, self.top_k)
        tp, fp, tn, fn = _multiclass_stat_scores_update(
            preds,
            target,
            self.num_classes,
            self.top_k,
            None,
            self.multidim_average,
            self.ignore_index,
        )
        mPrecision = _precision_recall_reduce(
            "precision",
            tp,
            fp,
            tn,
            fn,
            average=None,
            multidim_average=self.multidim_average,
            top_k=self.top_k,
            zero_division=self.zero_division,
        )

        match self.multidim_average:
            case "global":
                self.samples += preds.shape[0]
                self.mPrecision_running += mPrecision * preds.shape[0]
            case "samplewise":
                self.samples += preds.shape[0]
                self.mPrecision_running += mPrecision.sum(dim=0)


class MultilabelMPrecision(MultilabelPrecision):
    """Calculates mPrecision for multilabel tasks."""

    mPrecision_running: Tensor
    samples: Tensor

    @override
    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: Optional[Literal["macro", "none"]] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ):
        zero_division = kwargs.pop("zero_division", 0)
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _multilabel_stat_scores_arg_validation(
                num_labels,
                threshold,
                average,
                multidim_average,
                ignore_index,
                zero_division,
            )
        self.num_labels = num_labels
        self.threshold = threshold
        self.average: Optional[Literal["macro", "none"]] = average
        self.multidim_average: Literal["global", "samplewise"] = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        self.add_state(
            "mPrecision_running",
            torch.zeros(num_labels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "samples", torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum"
        )

    @override
    def compute(self) -> Tensor:
        avg = self.mPrecision_running / self.samples
        return avg

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.validate_args:
            _multilabel_stat_scores_tensor_validation(
                preds,
                target,
                self.num_labels,
                self.multidim_average,
                self.ignore_index,
            )

        preds, target = _multilabel_stat_scores_format(
            preds, target, self.num_labels, self.threshold, self.ignore_index
        )
        tp, fp, tn, fn = _multilabel_stat_scores_update(
            preds,
            target,
            self.multidim_average,
        )
        mPrecision = _precision_recall_reduce(
            "precision",
            tp,
            fp,
            tn,
            fn,
            average=None,
            multidim_average=self.multidim_average,
            multilabel=True,
            zero_division=self.zero_division,
        )

        match self.multidim_average:
            case "global":
                self.samples += preds.shape[0]
                self.mPrecision_running += mPrecision * preds.shape[0]
            case "samplewise":
                self.samples += preds.shape[0]
                self.mPrecision_running += mPrecision.sum(dim=0)


class MulticlassMRecall(MulticlassRecall):
    """Compute `mRecall` for multiclass tasks."""

    mRecall_running: Tensor
    samples: Tensor

    @override
    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        average: Optional[Literal["macro", "none"]] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ):
        zero_division = kwargs.pop("zero_division", 0)
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _multiclass_stat_scores_arg_validation(
                num_classes,
                top_k,
                average,
                multidim_average,
                ignore_index,
                zero_division,
            )
        self.num_classes = num_classes
        self.top_k = top_k
        self.average: Optional[Literal["macro", "none"]] = average
        self.multidim_average: Literal["global", "samplewise"] = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        self.add_state(
            "mRecall_running",
            torch.zeros(num_classes, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "samples", torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum"
        )

    @override
    def compute(self) -> Tensor:
        avg = self.mRecall_running / self.samples
        return avg

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.validate_args:
            _multiclass_stat_scores_tensor_validation(
                preds,
                target,
                self.num_classes,
                self.multidim_average,
                self.ignore_index,
            )

        preds, target = _multiclass_stat_scores_format(preds, target, self.top_k)
        tp, fp, tn, fn = _multiclass_stat_scores_update(
            preds,
            target,
            self.num_classes,
            self.top_k,
            None,
            self.multidim_average,
            self.ignore_index,
        )
        mRecall = _precision_recall_reduce(
            "recall",
            tp,
            fp,
            tn,
            fn,
            average=None,
            multidim_average=self.multidim_average,
            top_k=self.top_k,
            zero_division=self.zero_division,
        )

        match self.multidim_average:
            case "global":
                self.samples += preds.shape[0]
                self.mRecall_running += mRecall * preds.shape[0]
            case "samplewise":
                self.samples += preds.shape[0]
                self.mRecall_running += mRecall.sum(dim=0)


class MultilabelMRecall(MultilabelRecall):
    """Calculates mRecall for multilabel tasks."""

    mRecall_running: Tensor
    samples: Tensor

    @override
    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: Optional[Literal["macro", "none"]] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ):
        zero_division = kwargs.pop("zero_division", 0)
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _multilabel_stat_scores_arg_validation(
                num_labels,
                threshold,
                average,
                multidim_average,
                ignore_index,
                zero_division,
            )
        self.num_labels = num_labels
        self.threshold = threshold
        self.average: Optional[Literal["macro", "none"]] = average
        self.multidim_average: Literal["global", "samplewise"] = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        self.add_state(
            "mRecall_running",
            torch.zeros(num_labels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "samples", torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum"
        )

    @override
    def compute(self) -> Tensor:
        avg = self.mRecall_running / self.samples
        return avg

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.validate_args:
            _multilabel_stat_scores_tensor_validation(
                preds,
                target,
                self.num_labels,
                self.multidim_average,
                self.ignore_index,
            )

        preds, target = _multilabel_stat_scores_format(
            preds, target, self.num_labels, self.threshold, self.ignore_index
        )
        tp, fp, tn, fn = _multilabel_stat_scores_update(
            preds,
            target,
            self.multidim_average,
        )
        mRecall = _precision_recall_reduce(
            "recall",
            tp,
            fp,
            tn,
            fn,
            average=None,
            multidim_average=self.multidim_average,
            multilabel=True,
            zero_division=self.zero_division,
        )

        match self.multidim_average:
            case "global":
                self.samples += preds.shape[0]
                self.mRecall_running += mRecall * preds.shape[0]
            case "samplewise":
                self.samples += preds.shape[0]
                self.mRecall_running += mRecall.sum(dim=0)
