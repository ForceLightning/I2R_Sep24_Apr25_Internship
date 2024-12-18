"""Utility functions for URR-augmented attention model."""

# Standard Library
from enum import Enum, auto

# PyTorch
import torch
from torch import Tensor


def calc_uncertainty(score: Tensor) -> Tensor:
    """Calculate uncertainty.

    Args:
        score: Rough segmentation mask.

    Returns:
        Tensor: Uncertainty map.

    """
    # seg shape: bs, obj_n, h, w
    score_top, _ = score.topk(k=2, dim=1)
    uncertainty = score_top[:, 0] / (score_top[:, 1] + 1e-8)  # bs, h, w
    uncertainty = torch.exp(1 - uncertainty).unsqueeze(1)  # bs, 1, h, w
    return uncertainty


class URRSource(Enum):
    """Source for generating the low level feature maps for URR."""

    O1 = auto()
    O3 = auto()


class UncertaintyMode(Enum):
    """What form of UR/URR to use."""

    UR = auto()
    URR = auto()
