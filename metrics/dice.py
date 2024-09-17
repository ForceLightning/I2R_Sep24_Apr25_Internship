from typing import Any, Literal, override

import torch
from torch import Tensor
from torchmetrics.segmentation import GeneralizedDiceScore
from torchmetrics.segmentation.generalized_dice import _generalized_dice_update
from torchmetrics.utilities.compute import _safe_divide


class GeneralizedDiceScoreVariant(GeneralizedDiceScore):
    class_occurrences: torch.Tensor
    score_running: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        per_class: bool = False,
        weight_type: Literal["square", "simple", "linear"] = "square",
        weighted_average: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(
            num_classes, include_background, per_class, weight_type, **kwargs
        )

        self.num_classes = num_classes if include_background else num_classes - 1

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

    @override
    def compute(self) -> torch.Tensor:
        if not self.weighted_average:
            return super().compute()

        class_distribution = self.class_occurrences.div(
            self.class_occurrences.sum()
            if self.include_background
            else self.class_occurrences[1:].sum()
        )
        if self.include_background:
            return (class_distribution @ self.score_running) / self.samples
        else:
            return (class_distribution[1:] @ self.score_running[1:]) / self.samples

    def compute_macro_avg(self) -> torch.Tensor:
        return (
            (self.score_running / self.samples).mean()
            if self.include_background
            else (self.score_running[1:] / self.samples).mean()
        )

    def compute_per_class(self) -> torch.Tensor:
        return self.score_running / self.samples

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
            class_distribution = self.class_occurrences.div(
                self.class_occurrences.sum()
                if self.include_background
                else self.class_occurrences[1:].sum()
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
