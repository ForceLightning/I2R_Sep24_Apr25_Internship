"""Implements the FLA-Net Dataset.

See doi:10.48550/arXiv.2310.01861
"""

from __future__ import annotations

# Standard Library
import os
from typing import Literal, override

# Image Libraries
import cv2
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE
from PIL import Image

# PyTorch
import torch
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import masks_to_boxes
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose

# State-of-the-Art (SOTA) code
from thirdparty.fla_net.Code.utils.video_dataloader3 import HeatmapGenerator

# First party imports
from dataset.dataset import (
    DefaultDatasetProtocol,
    DefaultTransformsMixin,
    concatenate_imgs,
    load_train_indices,
)
from utils.types import ClassificationMode, LoadingMode


class FLANetDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]],
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
    def __getitem__(
        self, index
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:

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
        centre_x, centre_y = bbox[0] + w // 2, bbox[1] + h // 2
        heatmap = self.heatmap([[centre_x, centre_y, 1]])
        heatmap = self.transform_heatmap(heatmap)

        match self.classification_mode:
            case ClassificationMode.MULTILABEL_MODE:
                # NOTE: This turns the problem into a multilabel segmentation problem.
                # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
                # bitwise or operations to adhere to those conditions.
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
                    lab_mask_one_hot[:, :, 3]
                )
                lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
                    lab_mask_one_hot[:, :, 2]
                )
                out_mask = lab_mask_one_hot.bool().permute(-1, 0, 1)

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
