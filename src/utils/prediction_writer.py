"""Module for writing the predicted images with masks to the output directory."""

from __future__ import annotations

# Standard Library
import os
from typing import Any, Literal, Sequence

# Third-Party
from tqdm.auto import tqdm

# Image Libraries
from PIL.Image import Image

# PyTorch
import lightning as L
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from torchvision.transforms.v2 import functional as v2f
from torchvision.utils import draw_segmentation_masks

# First party imports
from utils.types import INV_NORM_RGB_DEFAULT, InverseNormalize, LoadingMode


class MaskImageWriter(BasePredictionWriter):
    """Writes the predicted images with masks to the output directory."""

    def __init__(
        self,
        loading_mode: LoadingMode,
        output_dir: str | None = None,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "epoch",
        inv_transform: InverseNormalize = INV_NORM_RGB_DEFAULT,
        format: Literal["apng", "tiff", "gif", "webp", "png"] = "gif",
    ):
        """Initialise the MaskImageWriter.

        Args:
            output_dir: The directory to save the images with masks.
            write_interval: The interval to write the images with masks.
            inv_transform: The inverse transform to apply to the images.
            loading_mode: The loading mode of the images.
            format: Output format of the images.

        """
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.inv_transform = inv_transform
        self.loading_mode = loading_mode
        self.format: Literal["apng", "tiff", "gif", "webp", "png"] = format
        if self.output_dir:
            if not os.path.exists(out_dir := os.path.normpath(self.output_dir)):
                os.makedirs(out_dir)

    def write_on_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        predictions: Sequence[tuple[torch.Tensor, torch.Tensor, list[str]]],
        batch_indices: Sequence[Any],
    ) -> None:
        """Save the predicted images with masks to the output directory.

        Args:
            trainer: The trainer object.
            pl_module: The lightning module.
            predictions: The predictions from the model.
            batch_indices: The indices of the batch.

        """
        if not self.output_dir:
            return
        for batched_mask_preds, batched_images, batched_fns in tqdm(
            predictions, desc="Batches"
        ):

            for mask_pred, image, fn in zip(
                batched_mask_preds, batched_images, batched_fns, strict=True
            ):
                num_frames = image.shape[0]

                masked_frames: list[Image] = []
                for frame in image:
                    masked_frame = _draw_masks(
                        frame, mask_pred, self.loading_mode, self.inv_transform
                    )
                    masked_frames.append(masked_frame)

                save_sample_fp = ".".join(fn.split(".")[:-1])

                save_path = os.path.join(
                    os.path.normpath(self.output_dir),
                    f"{save_sample_fp}_pred.{self.format}",
                )
                match self.format:
                    case "tiff":
                        masked_frames[0].save(
                            save_path,
                            append_images=masked_frames[1:],
                            save_all=True,
                        )
                    case "apng":
                        masked_frames[0].save(
                            save_path,
                            append_images=masked_frames[1:],
                            save_all=True,
                            duration=1000 // num_frames,
                            default_image=False,
                            disposal=1,
                            loop=0,
                        )
                    case "gif":
                        masked_frames[0].save(
                            save_path,
                            append_images=masked_frames[1:],
                            save_all=True,
                            duration=1000 // num_frames,
                            disposal=2,
                            loop=0,
                        )
                    case "webp":
                        masked_frames[0].save(
                            save_path,
                            append_images=masked_frames[1:],
                            save_all=True,
                            duration=1000 // num_frames,
                            loop=0,
                            background=(0, 0, 0, 0),
                            allow_mixed=True,
                        )
                    case "png":
                        for i, frame in enumerate(masked_frames):
                            save_path = os.path.join(
                                os.path.normpath(self.output_dir),
                                save_sample_fp,
                            )
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            save_path = os.path.join(
                                save_path, f"pred_{i:04d}.{self.format}"
                            )
                            frame.save(save_path)


def get_output_dir_from_ckpt_path(ckpt_path: str | None):
    """Get the output directory from the checkpoint path."""
    # Checkpoint paths are in the format:
    # ./checkpoints/<model type>/lightning_logs/<experiment name>/<version>/checkpoints/
    # <ckpt name>.ckpt
    if not ckpt_path:
        return None
    path = os.path.normpath(ckpt_path)
    split_path = path.split(os.sep)
    path_to_version = os.path.join(*split_path[:-2])
    return os.path.join(path_to_version, "predictions")


def _draw_masks(
    img: torch.Tensor,
    mask_one_hot: torch.Tensor,
    loading_mode: LoadingMode,
    inv_transform: InverseNormalize,
) -> Image:
    """Draws the masks on the image.

    Args:
        img: The image tensor.
        mask_one_hot: The one-hot encoded mask tensor.
        loading_mode: Whether the image is loaded as RGB or Greyscale.
        inv_transform: Inverse normalisation transformation of the image.

    Return:
        Image: The image with the masks drawn on it.

    """
    if loading_mode == LoadingMode.GREYSCALE:
        norm_img = inv_transform(img.repeat(3, 1, 1)).clamp(0, 1)
    else:
        norm_img = inv_transform(img).clamp(0, 1)

    return v2f.to_pil_image(
        draw_segmentation_masks(
            norm_img,
            mask_one_hot.bool(),
            colors=["black", "red", "blue", "green"],
            alpha=0.5,
        ),
        mode="RGB",
    )
