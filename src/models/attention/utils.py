"""Helper classes and typedefs for the residual frames-based attention models."""

from typing import Literal

REDUCE_TYPES = Literal["sum", "prod", "cat", "weighted", "weighted_learnable"]
