# -*- coding: utf-8 -*-
"""Dice score metrics."""
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
    class_weights: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        per_class: bool = False,
        weight_type: Literal["square", "simple", "linear"] = "square",
        weighted_average: bool = False,
        only_for_classes: list[bool] | list[int] | None = None,
        return_type: Literal["weighted_avg", "macro_avg", "per_class"] = "weighted_avg",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes, include_background, per_class, weight_type, **kwargs
        )

        self.num_classes = num_classes if include_background else num_classes - 1
        if only_for_classes:
            assert len(only_for_classes) == num_classes
            self.class_weights = torch.as_tensor(only_for_classes, dtype=torch.float32)
        else:
            self.class_weights = torch.ones((num_classes), dtype=torch.float32)

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
                "weighted_avg_metric", default=torch.zeros(1), dist_reduce_fx="sum"
            )
            self.add_state(
                "per_class_metric",
                default=torch.zeros(num_classes, dtype=torch.int32),
                dist_reduce_fx="sum",
            )

    @override
    def compute(self) -> torch.Tensor:
        if not self.weighted_average:
            return super().compute()

        self.class_weights = self.class_weights.to(self.class_occurrences.device)

        class_occurrences = (
            self.class_occurrences.to(torch.float32) * self.class_weights
        )

        class_distribution = _safe_divide(
            class_occurrences,
            (
                class_occurrences.sum()
                if self.include_background
                else class_occurrences[1:].sum()
            ),
        )

        self.weighted_avg_metric = (
            _safe_divide(class_distribution @ self.score_running, self.samples)
            if self.include_background
            else _safe_divide(
                class_distribution[1:] @ self.score_running[1:], self.samples
            )
        )

        # Set params
        self.macro_avg_metric = self._compute_macro_avg()
        self.per_class_metric = self._compute_per_class()

        match self.return_type:
            case "weighted_avg":
                return self.weighted_avg_metric
            case "macro_avg":
                return self.macro_avg_metric
            case "per_class":
                return self.per_class_metric

    def _compute_macro_avg(self) -> torch.Tensor:
        score = (self.score_running * self.class_weights)[self.class_weights == 1.0]
        score = score[1:] if not self.include_background else score
        score = _safe_divide(score, self.samples).mean()

        return score

    def _compute_per_class(self) -> torch.Tensor:
        score = self.score_running * self.class_weights
        score = _safe_divide(score, self.samples)

        return score

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        numerator, denominator = _generalized_dice_update(
            preds,
            target,
            self.num_classes,
            True,
            self.weight_type,  # pyright: ignore[reportArgumentType]
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
