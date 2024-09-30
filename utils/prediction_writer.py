from __future__ import annotations

import os
from typing import Any, Literal, Sequence

import lightning as L
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from PIL.Image import Image
from torchvision.transforms.v2 import functional as v2f
from torchvision.utils import draw_segmentation_masks

from utils.utils import InverseNormalize, LoadingMode


class MaskImageWriter(BasePredictionWriter):
    """Writes the predicted images with masks to the output directory.

    Args:
        output_dir: The directory to save the images with masks.
        write_interval: The interval to write the images with masks.
        inv_transform: The inverse transform to apply to the images.
        loading_mode: The loading mode of the images.
    """

    default_inv_transform = InverseNormalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )

    def __init__(
        self,
        output_dir: str,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "epoch",
        inv_transform: InverseNormalize = default_inv_transform,
        loading_mode: LoadingMode = LoadingMode.RGB,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.inv_transform = inv_transform
        self.loading_mode = loading_mode

    def write_on_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        predictions: Sequence[tuple[torch.Tensor, torch.Tensor, list[str]]],
        batch_indices: Sequence[Any],
    ) -> None:
        """Saves the predicted images with masks to the output directory.

        Args:
            trainer: The trainer object.
            pl_module: The lightning module.
            predictions: The predictions from the model.
            batch_indices: The indices of the batch.
        """

        for batched_mask_preds, batched_images, batched_fns in predictions:
            for mask_preds, images, fns in zip(
                batched_mask_preds, batched_images, batched_fns, strict=True
            ):
                for mask_pred, image, fn in zip(mask_preds, images, fns, strict=True):
                    masked_frames: list[Image] = []
                    for frame in image:
                        masked_frame = _draw_masks(frame, mask_pred, self.loading_mode)
                        masked_frames.append(masked_frame)
                    save_path = os.path.join(self.output_dir, f"{fn}_pred")
                    masked_frames[0].save(
                        save_path,
                        format="tiff",
                        append_images=masked_frames[1:],
                        save_all=True,
                    )


def _draw_masks(
    img: torch.Tensor, mask_one_hot: torch.Tensor, loading_mode: LoadingMode
) -> Image:
    """Draws the masks on the image.

    Args:
        img: The image tensor.
        mask_one_hot: The one-hot encoded mask tensor.

    Return:
        Image: The image with the masks drawn on it.
    """
    match loading_mode:
        case LoadingMode.GREYSCALE:
            img = img.repeat(3, 1, 1)
        case _:
            pass

    return v2f.to_pil_image(
        draw_segmentation_masks(
            img.clamp(0, 1),
            mask_one_hot,
            colors=["black", "red", "blue", "green"],
            alpha=0.5,
        ),
        mode="RGB",
    )
