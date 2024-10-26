"""Helper classes and typedefs for the residual frames-based attention models."""

from enum import Enum, auto
from typing import Literal


class ModelType(Enum):
    """Model architecture types."""

    UNET = auto()
    UNET_PLUS_PLUS = auto()
    TRANS_UNET = auto()


def get_model_type(enum_str: str):
    """Get enum from input string.

    Args:
        enum_str: String to match with enum.

    Return:
        ModelType: Resultant enum variant.

    Raises:
        KeyError: If the variant requested is not found.

    """
    return ModelType[enum_str.upper()]


REDUCE_TYPES = Literal["sum", "prod", "cat", "weighted", "weighted_learnable"]
