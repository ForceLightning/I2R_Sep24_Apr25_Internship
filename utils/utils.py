from typing import Sequence

import torch
from torchvision.transforms import Normalize


class InverseNormalize(Normalize):
    """Inverses the normalization and returns the reconstructed images in the input."""

    def __init__(
        self,
        mean: Sequence[float | int],
        std: Sequence[float | int],
    ):
        mean_tensor = torch.as_tensor(mean)
        std_tensor = torch.as_tensor(std)
        std_inv = 1 / (std_tensor + 1e-7)
        mean_inv = -mean_tensor * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return super().__call__(tensor.clone())
