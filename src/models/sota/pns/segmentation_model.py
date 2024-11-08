"""Segmentation Model implementation for PNSnet."""

# Standard Library
from typing import override

# Third-Party
from segmentation_models_pytorch.base.model import SegmentationModel

# PyTorch
from torch import Tensor
from torch.nn.common_types import _size_2_t

# First party imports
from models.sota.pns.pns import PNSNet


class PNSSegModel(PNSNet, SegmentationModel):
    """SegmentationModel wrapper for PNSNet."""

    @override
    def __init__(self, image_shape: _size_2_t, classes: int = 1, num_frames: int = 6):
        super().__init__(image_shape, classes, num_frames)

    @override
    def initialize(self):
        pass

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Input shape: (B, F, C, H, W)
        # Output shape: (B, F-1, classes, H, W)
        out = super().forward(x)

        # Take the last frame's output.
        return out[:, -1, :, :, :]
