"""Implements the FLA-Net Dataset.

See doi:10.48550/arXiv.2310.01861
"""

from __future__ import annotations

# Standard Library
import os
from typing import Literal, Optional, Sequence, override

# Scientific Libraries
import numpy as np
from numpy import typing as npt

# Image Libraries
import cv2
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE
from PIL import Image

# PyTorch
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import masks_to_boxes
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose

# First party imports
from dataset.dataset import (
    DefaultDatasetProtocol,
    DefaultTransformsMixin,
    concatenate_imgs,
    load_train_indices,
)
from utils.types import ClassificationMode, LoadingMode


class HeatmapGenerator:
    """Heatmap generator for FLA-Net."""

    def __init__(
        self, output_res: int, num_joints: int, sigma: Optional[int] = None
    ) -> None:
        """Initialise the Heatmap generator."""
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma is None:
            self.sigma = self.output_res / 64
        else:
            self.sigma = sigma

        size = 6 * self.sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * self.sigma + 1, 3 * self.sigma + 1
        self.g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma**2))

    def __call__(
        self, joints: Sequence[tuple[int, int, int]]
    ) -> npt.NDArray[np.float32]:
        """Generate heatmap."""
        hms = np.zeros(
            (self.num_joints, self.output_res, self.output_res), dtype=np.float32
        )
        for idx, pt in enumerate(joints):
            if pt[2] > 0:
                x, y = int(pt[0]), int(pt[1])
                if any((x < 0, y < 0, x >= self.output_res, y >= self.output_res)):
                    continue

                ul = int(np.round(x - 3 * self.sigma - 1)), int(
                    np.round(y - 3 * self.sigma - 1)
                )
                br = int(np.round(x + 3 * self.sigma + 2)), int(
                    np.round(y + 3 * self.sigma + 2)
                )

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)

                hms[idx, aa:bb, cc:dd] = np.maximum(
                    hms[idx, aa:bb, cc:dd], self.g[a:b, c:d]
                )
        return hms


class FLANetDataset(
    Dataset[tuple[Tensor, Tensor, Tensor, str]],
    DefaultTransformsMixin,
    DefaultDatasetProtocol,
):
    """Dataset for the FLA-Net implementation."""

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        idxs_dir: str,
        frames: int,
        select_frame_method: Literal["consecutive", "specific"],
        transform_img: Compose,
        transform_mask: Compose,
        transform_heatmap: Compose | None = None,
        transform_together: Compose | None = None,
        batch_size: int = 8,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
        image_size: _size_2_t = (224, 224),
    ) -> None:
        """Initialise the dataset for the FLA-Net implementation."""
        super().__init__()

        self.frames = frames
        self.select_frame_method: Literal["consecutive", "specific"] = (
            select_frame_method
        )

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        height = image_size[0] if isinstance(image_size, tuple) else image_size
        width = image_size[1] if isinstance(image_size, tuple) else image_size
        self.image_size: tuple[int, int] = (height, width)

        self.img_list = os.listdir(self.img_dir)
        self.mask_list = os.listdir(self.mask_dir)

        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.transform_heatmap = (
            transform_heatmap
            if transform_heatmap
            else Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])
        )
        self.transform_together = (
            transform_together if transform_together else Compose([v2.Identity()])
        )

        self.train_idxs: list[int]
        self.valid_idxs: list[int]

        self.batch_size = batch_size
        if mode != "test" and not combine_train_val:
            load_train_indices(
                self,
                os.path.join(idxs_dir, "train_indices.pkl"),
                os.path.join(idxs_dir, "val_indices.pkl"),
            )

        self.classification_mode = classification_mode
        self.loading_mode = loading_mode
        self._imread_mode = (
            IMREAD_COLOR if self.loading_mode == LoadingMode.RGB else IMREAD_GRAYSCALE
        )
        self.heatmap = HeatmapGenerator(224, 1, 10)

    @override
    def __getitem__(self, index) -> tuple[Tensor, Tensor, Tensor, str]:

        # Define Cine file name
        img_name: str = self.img_list[index]
        mask_name: str = self.img_list[index].split(".")[0] + ".nii.png"

        # PERF: Initialise the output tensor ahead of time to reduce memory allocation
        # time.
        _, img_list = cv2.imreadmulti(
            os.path.join(self.img_dir, img_name), flags=IMREAD_GRAYSCALE
        )
        combined_video = torch.empty((30, *self.image_size), dtype=torch.uint8)
        for i in range(30):
            img = img_list[i]
            img = cv2.resize(img, self.image_size)
            combined_video[i, :, :] = torch.as_tensor(img)

        combined_video = combined_video.view(30, 1, *self.image_size)
        combined_video = self.transform_img(combined_video)

        with Image.open(
            os.path.join(self.mask_dir, mask_name), formats=["png"]
        ) as mask:
            out_mask = tv_tensors.Mask(self.transform_mask(mask))

        lab_mask_one_hot = F.one_hot(
            out_mask.clone().squeeze(), num_classes=4
        )  # H x W x C
        lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
            lab_mask_one_hot[:, :, 3]
        )
        lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
            lab_mask_one_hot[:, :, 2]
        )
        out_mask_one_hot = tv_tensors.Mask(lab_mask_one_hot.bool().permute(-1, 0, 1))

        class_mask = (
            out_mask_one_hot[1, :, :]
            .bitwise_or(out_mask_one_hot[2, :, :])
            .bitwise_or(out_mask_one_hot[3, :, :])
            .unsqueeze(0)
        )
        bbox = masks_to_boxes(class_mask).squeeze()
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        centre_x, centre_y = int(bbox[0] + w // 2), int(bbox[1] + h // 2)
        heatmap = self.heatmap([(centre_x, centre_y, 1)])
        heatmap = self.transform_heatmap(heatmap)

        match self.classification_mode:
            case ClassificationMode.MULTILABEL_MODE:
                # NOTE: This turns the problem into a multilabel segmentation problem.
                # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
                # bitwise or operations to adhere to those conditions.
                out_mask = out_mask_one_hot

            case ClassificationMode.MULTICLASS_MODE:
                pass
            case _:
                raise NotImplementedError(
                    f"The mode {self.classification_mode.name} is not implemented"
                )

        out_video, out_mask, out_heatmap = self.transform_together(
            combined_video, out_mask, heatmap.permute(1, 0, 2)
        )
        out_video = concatenate_imgs(self.frames, self.select_frame_method, out_video)

        return out_video, out_mask.squeeze().long(), out_heatmap, img_name

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.img_list)
