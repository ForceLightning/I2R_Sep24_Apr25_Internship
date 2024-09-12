from typing import Any, Literal, override

import torch
from torch import Tensor
from torchmetrics.segmentation import GeneralizedDiceScore
from torchmetrics.segmentation.generalized_dice import _generalized_dice_update
from torchmetrics.utilities.compute import _safe_divide


class GeneralizedDiceScoreVariant(GeneralizedDiceScore):
    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        per_class: bool = False,
        weight_type: Literal["square", "simple", "linear"] = "square",
        **kwargs: Any
    ) -> None:
        super().__init__(
            num_classes, include_background, per_class, weight_type, **kwargs
        )

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        numerator, denominator = _generalized_dice_update(
            preds, target, self.num_classes, self.include_background, self.weight_type
        )
        self.score += _generalized_dice_compute(
            numerator, denominator, self.per_class
        ).sum(dim=0)
        self.samples += preds.shape[0]


def _generalized_dice_compute(
    numerator: Tensor, denominator: Tensor, per_class: bool = True
) -> Tensor:
    """Overrides the default computation by setting undefined behaviour to return `1.0`"""
    if not per_class:
        numerator = torch.sum(numerator, 1)
        denominator = torch.sum(denominator, 1)
    return _safe_divide(numerator, denominator, 1.0)
