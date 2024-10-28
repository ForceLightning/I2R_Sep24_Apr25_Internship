"""Helper module containing type and class definitions."""

from enum import Enum, auto
from typing import Sequence, override

import torch
from torchvision.transforms import v2


class InverseNormalize(v2.Normalize):
    """Inverses the normalization and returns the reconstructed images in the input."""

    def __init__(
        self,
        mean: Sequence[float | int],
        std: Sequence[float | int],
    ):
        """Initialise the InverseNormalize class.

        Args:
            mean: The mean value for the normalisation.
            std: The standard deviation value for the normalisation.

        """
        mean_tensor = torch.as_tensor(mean)
        std_tensor = torch.as_tensor(std)
        std_inv = 1 / (std_tensor + 1e-7)
        mean_inv = -mean_tensor * std_inv
        super().__init__(mean=mean_inv.tolist(), std=std_inv.tolist())

    @override
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return super().__call__(tensor.clone())


INV_NORM_RGB_DEFAULT = InverseNormalize(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)
INV_NORM_GREYSCALE_DEFAULT = InverseNormalize(mean=[0.449], std=[0.226])


class ClassificationMode(Enum):
    """The classification mode for the model.

    MULTICLASS_MODE: The model is trained to predict a single class for each pixel.
    MULTILABEL_MODE: The model is trained to predict multiple classes for each pixel.
    """

    MULTICLASS_MODE = auto()
    MULTILABEL_MODE = auto()


class ResidualMode(Enum):
    """The residual frame calculation mode for the model.

    SUBTRACT_NEXT_FRAME: Subtracts the next frame from the current frame.
    OPTICAL_FLOW_CPU: Calculates the optical flow using the CPU.
    OPTICAL_FLOW_GPU: Calculates the optical flow using the GPU.
    """

    SUBTRACT_NEXT_FRAME = auto()
    OPTICAL_FLOW_CPU = auto()
    OPTICAL_FLOW_GPU = auto()


class LoadingMode(Enum):
    """Determines the image loading mode for the dataset.

    RGB: The images are loaded in RGB mode.
    GREYSCALE: The images are loaded in greyscale mode.
    """

    RGB = auto()
    GREYSCALE = auto()


class ModelType(Enum):
    """Model architecture types."""

    UNET = auto()
    UNET_PLUS_PLUS = auto()
    TRANS_UNET = auto()
