"""Implements the Vivim dataset.

See doi: 10.48550/arXiv.2401.14168
"""

from __future__ import annotations

import os
from typing import Literal, override

import cv2
import numpy as np
import torch
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE
from numpy import typing as npt
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose

from dataset.dataset import (
    DefaultDatasetProtocol,
    DefaultTransformsMixin,
    concatenate_imgs,
    load_train_indices,
)
from utils.types import ClassificationMode, LoadingMode


class VivimDataset(
    Dataset[tuple[Tensor, Tensor, str] | tuple[Tensor, Tensor, Tensor, str]],
    DefaultTransformsMixin,
    DefaultDatasetProtocol,
):
    """Dataset for the Vivim implementation."""

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        idxs_dir: str,
        transform_img: Compose,
        transform_mask: Compose,
        transform_together: Compose | None = None,
        frames: int = 30,
        select_frame_method: Literal["consecutive", "specific"] = "consecutive",
        batch_size: int = 4,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
        with_edge: bool = False,
    ) -> None:
        """Initialise the dataset for the Vivim implementation.

        Args:
            img_dir: Path to the directory containing the images.
            mask_dir: Path to the directory containing the masks.
            idxs_dir: Path to the directory containing the indices.
            transform_img: Transform to apply to the images.
            transform_mask: Transform to apply to the masks.
            transform_together: The transform to apply to both the images and masks.
            frames: Number of frames to use for the model (out of 30).
            select_frame_method: How to select the frames (if fewer than 30).
            batch_size: Batch size for the dataset.
            mode: Runtime mode.
            classification_mode: Classification mode for the dataset.
            loading_mode: Determines the cv2.imread flags for the images.
            combine_train_val: Whether to combine the train/val sets.
            with_edge: Whether to return edgemaps.

        Raises:
            NotImplementedError: If the classification mode is not implemented.
            RuntimeError: If the indices fail to load.
            AssertionError: If the indices are not disjoint.

        """
        super().__init__()

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_list: list[str] = os.listdir(self.img_dir)
        self.mask_list: list[str] = os.listdir(self.mask_dir)

        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.transform_together = (
            transform_together if transform_together else Compose([v2.Identity()])
        )

        self.frames = frames
        self.select_frame_method: Literal["consecutive", "specific"] = (
            select_frame_method
        )
        self.train_idxs: list[int]
        self.valid_idxs: list[int]

        self.batch_size = batch_size
        self.mode = mode
        self.classification_mode = classification_mode
        self.loading_mode = loading_mode
        self._imread_mode = (
            IMREAD_COLOR if self.loading_mode == LoadingMode.RGB else IMREAD_GRAYSCALE
        )

        if mode != "test" and not combine_train_val:
            load_train_indices(
                self,
                os.path.join(idxs_dir, "train_indices.pkl"),
                os.path.join(idxs_dir, "val_indices.pkl"),
            )

        self.with_edge = with_edge

    @override
    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, Tensor, str] | tuple[Tensor, Tensor, Tensor, str]:
        # Define Cine file name
        img_name: str = self.img_list[index]
        mask_name: str = self.img_list[index].split(".")[0] + ".nii.png"

        # PERF: Initialise the output tensor ahead of time to reduce memory allocation
        # time.
        _, img_list = cv2.imreadmulti(
            os.path.join(self.img_dir, img_name), flags=IMREAD_GRAYSCALE
        )
        combined_video = torch.empty((30, 224, 224), dtype=torch.uint8)
        for i in range(30):
            img = img_list[i]
            img = cv2.resize(img, (224, 224))
            combined_video[i, :, :] = torch.as_tensor(img)

        combined_video = combined_video.view(30, 1, 224, 224)
        combined_video = self.transform_img(combined_video)

        with Image.open(
            os.path.join(self.mask_dir, mask_name), formats=["png"]
        ) as mask:
            out_mask = tv_tensors.Mask(self.transform_mask(mask))

        lab_mask_one_hot = F.one_hot(out_mask.squeeze(), num_classes=4)  # H x W x C

        match self.classification_mode:
            case ClassificationMode.MULTILABEL_MODE:
                # NOTE: This turns the problem into a multilabel segmentation problem.
                # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
                # bitwise or operations to adhere to those conditions.
                lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
                    lab_mask_one_hot[:, :, 3]
                )
                lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
                    lab_mask_one_hot[:, :, 2]
                )
                out_mask = tv_tensors.Mask(lab_mask_one_hot.bool().permute(-1, 0, 1))

            case ClassificationMode.MULTICLASS_MODE:
                pass
            case _:
                raise NotImplementedError(
                    f"The mode {self.classification_mode.name} is not implemented"
                )

        out_video, out_mask = self.transform_together(combined_video, out_mask)
        out_video = concatenate_imgs(self.frames, self.select_frame_method, out_video)
        out_mask = out_mask.squeeze().long()

        if self.with_edge:
            edgemap = self._onehot_to_binary_edges(
                lab_mask_one_hot.permute(-1, 0, 1).numpy(), 2, 2
            )
            edgemap = edgemap.reshape(1, *edgemap.shape)

            return out_video, out_mask, edgemap, img_name

        return out_video, out_mask, img_name

    def _onehot_to_binary_edges(
        self, mask: npt.NDArray[np.float32], radius: int, num_classes: int
    ) -> Tensor:
        mask_pad = np.pad(
            mask, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0
        )
        edgemap = np.zeros(mask.shape[1:])
        for i in range(num_classes):
            a = distance_transform_edt(mask_pad[i, :])
            b = distance_transform_edt(1.0 - mask_pad[i, :])
            assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
            dist = a + b
            dist = dist[1:-1, 1:-1]
            dist[dist > radius] = 0
            edgemap += dist

        edgemap = (edgemap > 0).astype(np.uint8)
        return torch.from_numpy(edgemap)

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.img_list)
