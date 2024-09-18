from typing import Any, Literal, override

import torch
from torch import Tensor
from torchmetrics.segmentation import GeneralizedDiceScore
from torchmetrics.segmentation.generalized_dice import _generalized_dice_update
from torchmetrics.utilities.compute import _safe_divide


class GeneralizedDiceScoreVariant(GeneralizedDiceScore):
    class_occurrences: torch.Tensor
    score_running: torch.Tensor
    macro_avg_metric: torch.Tensor
    per_class_metric: torch.Tensor
    weighted_avg_metric: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        per_class: bool = False,
        weight_type: Literal["square", "simple", "linear"] = "square",
        weighted_average: bool = False,
        return_type: Literal["weighted_avg", "macro_avg", "per_class"] = "weighted_avg",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes, include_background, per_class, weight_type, **kwargs
        )

        self.num_classes = num_classes if include_background else num_classes - 1
        self.return_type: Literal["weighted_avg", "macro_avg", "per_class"] = (
            return_type
        )

        self.weighted_average = weighted_average
        if self.weighted_average:
            self.add_state(
                "class_occurrences",
                default=torch.zeros(num_classes, dtype=torch.int32),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "score_running",
                default=torch.zeros(num_classes),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "weighted_avg_metric", default=torch.zeros(1), dist_reduce_fx="mean"
            )
            self.add_state(
                "per_class_metric",
                default=torch.zeros(num_classes, dtype=torch.int32),
                dist_reduce_fx="mean",
            )

    @override
    def compute(self) -> torch.Tensor:
        if not self.weighted_average:
            return super().compute()

        class_distribution = _safe_divide(
            self.class_occurrences,
            (
                self.class_occurrences.sum()
                if self.include_background
                else self.class_occurrences[1:].sum()
            ),
        )

        self.macro_avg_metric = self._compute_macro_avg()
        self.per_class_metric = self._compute_per_class()
        self.weighted_avg_metric = (
            _safe_divide(class_distribution @ self.score_running, self.samples)
            if self.include_background
            else _safe_divide(
                class_distribution[1:] @ self.score_running[1:], self.samples
            )
        )
        match self.return_type:
            case "weighted_avg":
                return self.weighted_avg_metric
            case "macro_avg":
                return self.macro_avg_metric
            case "per_class":
                return self.per_class_metric

    def _compute_macro_avg(self) -> torch.Tensor:
        score = (
            _safe_divide(self.score_running, self.samples).mean()
            if self.include_background
            else _safe_divide(self.score_running[1:], self.samples).mean()
        )

        return score

    def _compute_per_class(self) -> torch.Tensor:
        score = _safe_divide(self.score_running, self.samples)

        return score

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        numerator, denominator = _generalized_dice_update(
            preds, target, self.num_classes, True, self.weight_type
        )
        dice = _generalized_dice_compute(numerator, denominator, self.per_class).sum(
            dim=0
        )
        self.samples += preds.shape[0]

        if self.weighted_average:
            self.score_running += dice
            class_occurrences = preds.sum(dim=[0, 2, 3])
            self.class_occurrences += class_occurrences
            class_distribution = _safe_divide(
                self.class_occurrences,
                (
                    self.class_occurrences.sum()
                    if self.include_background
                    else self.class_occurrences[1:].sum()
                ),
            )
            if self.include_background:
                self.score = dice @ class_distribution
            else:
                self.score = dice[1:] @ class_distribution[1:]
        else:
            self.score += dice


def _generalized_dice_compute(
    numerator: Tensor, denominator: Tensor, per_class: bool = True
) -> Tensor:
    """Overrides the default computation by setting undefined behaviour to return `1.0`"""
    if not per_class:
        numerator = torch.sum(numerator, 1)
        denominator = torch.sum(denominator, 1)
    return _safe_divide(numerator, denominator, 1.0)
