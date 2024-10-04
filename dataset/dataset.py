# -*- coding: utf-8 -*-
"""Module for the dataset classes and functions for the cardiac MRI images."""
from __future__ import annotations

import os
import pickle
import random
from typing import Literal, Sequence, override

import numpy as np
import PIL.ImageSequence as ImageSequence
import torch
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE
from cv2 import typing as cvt
from numpy import typing as npt
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    SubsetRandomSampler,
    default_collate,
)
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose

from utils.utils import ClassificationMode, LoadingMode

SEED_CUS = 1  # RNG seed.


class LGEDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        idxs_dir: str,
        transform_img: Compose,
        transform_mask: Compose,
        transform_together: Compose | None = None,
        batch_size: int = 8,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
    ) -> None:
        """LGE dataset for the cardiac LGE MRI images.

        Args:
            img_dir: The directory containing the LGE images.
            mask_dir: The directory containing the masks for the LGE images.
            idxs_dir: The directory containing the indices for the training and
            validation sets.
            transform_img: The transform to apply to the images.
            transform_mask: The transform to apply to the masks.
            transform_together: The transform to apply to both the images and masks.
            batch_size: The batch size for the dataset.
            mode: The mode of the dataset.
            classification_mode: The classification mode for the dataset.
            loading_mode: Determines the cv2.imread flags for the images.
            combine_train_val: Whether to combine the train/val sets.

        Raises:
            NotImplementedError: If the classification mode is not implemented.
            RuntimeError: If the indices fail to load.
            AssertionError: If the indices are not disjoint.
        """
        super().__init__()

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_list = os.listdir(self.img_dir)
        self.mask_list = os.listdir(self.mask_dir)

        self.transform_img = transform_img
        self.transform_mask = transform_mask
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

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Gets a batch of images, masks, and the image names from the dataset.

        Args:
            index: The index of the batch.

        Returns:
            tuple[torch.Tensor, torch.Tensor | npt.NDArray[np.floating[Any]], str]: The
            images, masks, and image names.

        Raises:
            ValueError: If the image is not in .PNG format.
            NotImplementedError: If the classification mode is not implemented
        """
        img_name = self.img_list[index]
        mask_name = self.img_list[index].split(".")[0] + "_0000.nii.png"

        # PERF(PIL): This reduces the loading and transform time by 60% when compared
        # to OpenCV.

        # Convert LGE to RGB or Greyscale
        with Image.open(os.path.join(self.img_dir, img_name), formats=["png"]) as img:
            img_list = (
                img.convert("RGB")
                if self.loading_mode == LoadingMode.RGB
                else img.convert("L")
            )
        out_img: torch.Tensor = self.transform_img(img_list)

        with Image.open(
            os.path.join(self.mask_dir, mask_name), formats=["png"]
        ) as mask:
            out_mask = tv_tensors.Mask(self.transform_mask(mask))

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
                out_mask = tv_tensors.Mask(lab_mask_one_hot.bool().permute(-1, 0, 1))

            case ClassificationMode.MULTICLASS_MODE:
                pass
            case _:
                raise NotImplementedError(
                    f"The mode {self.classification_mode.name} is not implemented"
                )

        out_img, out_mask = self.transform_together(out_img, out_mask)

        return out_img, out_mask.squeeze().long(), img_name

    def __len__(self) -> int:
        return len(self.img_list)


class CineDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
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
    ) -> None:
        """Dataset for the Cine baseline implementation

        Args:
            img_dir: Path to the directory containing the images.
            mask_dir: Path to the directory containing the masks.
            idxs_dir: Path to the directory containing the indices.
            transform_img: Transform to apply to the images.
            transform_mask: Transform to apply to the masks.
            transform_together: The transform to apply to both the images and masks.
            batch_size: Batch size for the dataset.
            mode: Runtime mode.
            classification_mode: Classification mode for the dataset.
            loading_mode: Determines the cv2.imread flags for the images.
            combine_train_val: Whether to combine the train/val sets.

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

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        # Define Cine file name
        img_name: str = self.img_list[index]
        mask_name: str = self.img_list[index].split(".")[0] + ".nii.png"

        # PERF(PIL): This reduces the loading and transform time by 60% when compared
        # to OpenCV.
        with Image.open(
            os.path.join(self.img_dir, img_name), formats=["tiff"]
        ) as img_pil:
            img_list = ImageSequence.all_frames(
                img_pil,
                lambda img: (
                    img.convert("RGB")
                    if self.loading_mode == LoadingMode.RGB
                    else img.convert("L")
                ),
            )

            img_list = self.transform_img(img_list)
            combined_imgs = tv_tensors.Video(default_collate(img_list))

        with Image.open(
            os.path.join(self.mask_dir, mask_name), formats=["png"]
        ) as mask:
            out_mask = tv_tensors.Mask(self.transform_mask(mask))

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
                out_mask = tv_tensors.Mask(lab_mask_one_hot.bool().permute(-1, 0, 1))

            case ClassificationMode.MULTICLASS_MODE:
                pass
            case _:
                raise NotImplementedError(
                    f"The mode {self.classification_mode.name} is not implemented"
                )

        out_video, out_mask = self.transform_together(combined_imgs, out_mask)
        out_video = concatenate_imgs(self.frames, self.select_frame_method, out_video)

        f, c, h, w = out_video.shape
        out_video = out_video.view(f * c, h, w)

        return out_video, out_mask.squeeze().long(), img_name

    def __len__(self) -> int:
        return len(self.img_list)


class TwoPlusOneDataset(CineDataset):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        idxs_dir: str,
        frames: int,
        select_frame_method: Literal["consecutive", "specific"],
        transform_img: Compose,
        transform_mask: Compose,
        transform_together: Compose | None = None,
        batch_size: int = 2,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
    ) -> None:
        """Cine dataset for the cardiac cine MRI images.

        Args:
            img_dir: The directory containing the CINE images.
            mask_dir: The directory containing the masks for the CINE images.
            idxs_dir: The directory containing the indices for the training and
                validation sets.
            frames: The number of frames to concatenate.
            select_frame_method: The method of selecting frames to concatenate.
            transform_img: The transform to apply to the images.
            transform_mask: The transform to apply to the masks.
            transform_together: The transform to apply to both the images and masks.
            batch_size: The batch size for the dataset.
            mode: The mode of the dataset.
            classification_mode: The classification mode for the dataset.
            loading_mode: Determines the cv2.imread flags for the images.
            combine_train_val: Whether to combine the train/val sets.

        Raises:
            NotImplementedError: If the classification mode is not implemented.
            RuntimeError: If the indices fail to load.
            AssertionError: If the indices are not disjoint.
        """
        self.frames = frames
        self.select_frame_method: Literal["consecutive", "specific"] = (
            select_frame_method
        )
        super().__init__(
            img_dir,
            mask_dir,
            idxs_dir,
            transform_img,
            transform_mask,
            transform_together,
            batch_size,
            mode,
            classification_mode,
            loading_mode=loading_mode,
            combine_train_val=combine_train_val,
        )

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        img_name = self.img_list[index]
        mask_name = self.img_list[index].split(".")[0] + ".nii.png"

        # PERF(PIL): This reduces the loading and transform time by 60% when compared
        # to OpenCV.
        with Image.open(
            os.path.join(self.img_dir, img_name), formats=["tiff"]
        ) as img_pil:
            img_list = ImageSequence.all_frames(
                img_pil,
                lambda img: (
                    img.convert("RGB")
                    if self.loading_mode == LoadingMode.RGB
                    else img.convert("L")
                ),
            )

            img_list = self.transform_img(img_list)

            combined_video = tv_tensors.Video(default_collate(img_list))

        # Perform necessary operations on the mask
        with Image.open(
            os.path.join(self.mask_dir, mask_name), formats=["png"]
        ) as mask:
            out_mask = tv_tensors.Mask(self.transform_mask(mask))

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

        combined_video, out_mask = self.transform_together(combined_video, out_mask)
        assert (
            len(combined_video.shape) == 4
        ), f"Combined images must be of shape: (F, C, H, W) but is {combined_video.shape} instead."

        out_video = concatenate_imgs(
            self.frames, self.select_frame_method, combined_video
        )

        return out_video, out_mask.squeeze().long(), img_name


class TwoStreamDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        lge_dir: str,
        cine_dir: str,
        mask_dir: str,
        idxs_dir: str,
        transform_img: Compose,
        transform_mask: Compose,
        transform_together: Compose | None = None,
        batch_size: int = 8,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
    ):
        """Two stream dataset for the cardiac LGE MRI images.

        Args:
            lge_dir: The directory containing the LGE images.
            cine_dir: The directory containing the CINE images.
            mask_dir: The directory containing the masks for the LGE images.
            idxs_dir: The directory containing the indices for the training and
            validation sets.
            transform_img: The transform to apply to the images.
            transform_mask: The transform to apply to the masks.
            transform_together: The transform to apply to both the images and masks.
            batch_size: The batch size for the dataset.
            mode: The mode of the dataset.
            classification_mode: The classification mode for the dataset.
            loading_mode: Determines the cv2.imread flags for the images.
            combine_train_val: Whether to combine the train/val sets.

        Raises:
            NotImplementedError: If the classification mode is not implemented.
            RuntimeError: If the indices fail to load.
            AssertionError: If the indices are not disjoint.
        """
        super().__init__()

        self.lge_dir = lge_dir
        self.cine_dir = cine_dir
        self.mask_dir = mask_dir

        self.lge_list = os.listdir(self.lge_dir)
        self.cine_list = os.listdir(self.cine_dir)
        self.mask_list = os.listdir(self.mask_dir)

        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.transform_together = (
            transform_together if transform_together else Compose([v2.Identity()])
        )

        self.batch_size = batch_size
        self.train_idxs: list[int]
        self.valid_idxs: list[int]

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

    @property
    def img_dir(self):
        return self.lge_dir

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Gets a batch of LGE images, CINE images, masks, and the image names from the
        dataset.

        Args:
            index: The index of the batch.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]: The LGE images, CINE
            images, masks, and image names.

        Raises:
            ValueError: If the image is not in .PNG format.
            NotImplementedError: If the classification mode is not implemented.
        """
        # Define names for all files using the same LGE base
        lge_name = self.lge_list[index]
        cine_name = self.lge_list[index].split(".")[0] + "_0000.nii.tiff"
        mask_name = self.lge_list[index].split(".")[0] + "_0000.nii.png"

        if not lge_name.endswith(".png"):
            raise ValueError("Invalid image type for file: {lge_name}")

        # PERF(PIL): This reduces the loading and transform time by 60% when compared
        # to OpenCV.

        # Convert LGE to RGB or Greyscale
        with Image.open(os.path.join(self.lge_dir, lge_name), formats=["png"]) as lge:
            out_lge = self.transform_img(lge.convert("L"))

        with Image.open(
            os.path.join(self.cine_dir, cine_name), formats=["tiff"]
        ) as cine:
            img_list = ImageSequence.all_frames(
                cine,
                lambda img: (
                    img.convert("RGB")
                    if self.loading_mode == LoadingMode.RGB
                    else img.convert("L")
                ),
            )
            img_list = self.transform_img(img_list)

            combined_cines = tv_tensors.Video(default_collate(img_list))

        out_lge.squeeze()

        with Image.open(
            os.path.join(self.mask_dir, mask_name), formats=["png"]
        ) as mask:
            mask_t = tv_tensors.Mask(self.transform_mask(mask))

        match self.classification_mode:
            case ClassificationMode.MULTILABEL_MODE:
                # NOTE: This turns the problem into a multilabel segmentation problem.
                # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
                # bitwise or operations to adhere to those conditions.
                lab_mask_one_hot = F.one_hot(
                    mask_t.squeeze(), num_classes=4
                )  # H x W x C
                lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
                    lab_mask_one_hot[:, :, 3]
                )
                lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
                    lab_mask_one_hot[:, :, 2]
                )
                mask_t = tv_tensors.Mask(lab_mask_one_hot.bool().permute(-1, 0, 1))

            case ClassificationMode.MULTICLASS_MODE:
                pass
            case _:
                raise NotImplementedError(
                    f"The mode {self.classification_mode.name} is not implemented"
                )

        # Perform transforms which must occur on all inputs together.
        out_lge, combined_cines, out_mask = self.transform_together(
            out_lge, combined_cines, mask_t
        )

        f, c, h, w = combined_cines.shape
        out_cine = combined_cines.reshape(f * c, h, w)

        # Combine the Cine channels.
        return out_lge, out_cine, out_mask.squeeze().long(), lge_name

    def __len__(self):
        return len(self.cine_list)


class ResidualTwoPlusOneDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]
):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        idxs_dir: str,
        frames: int,
        select_frame_method: Literal["consecutive", "specific"],
        transform_img: Compose,
        transform_mask: Compose,
        transform_together: Compose | None = None,
        batch_size: int = 2,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
    ) -> None:
        super().__init__()
        self.frames = frames
        self.select_frame_method: Literal["consecutive", "specific"] = (
            select_frame_method
        )

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_list: list[str] = os.listdir(self.img_dir)
        self.mask_list: list[str] = os.listdir(self.mask_dir)

        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.transform_together = (
            transform_together if transform_together else Compose([v2.Identity()])
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

    def __len__(self) -> int:
        return len(self.img_list)

    @override
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        img_name = self.img_list[index]
        mask_name = self.img_list[index].split(".")[0] + ".nii.png"

        with Image.open(
            os.path.join(self.img_dir, img_name), formats=["tiff"]
        ) as img_pil:
            img_list = ImageSequence.all_frames(
                img_pil,
                lambda img: (
                    img.convert("RGB")
                    if self.loading_mode == LoadingMode.RGB
                    else img.convert("L")
                ),
            )
            img_list = self.transform_img(img_list)

            combined_video = tv_tensors.Video(default_collate(img_list))

        # Perform necessary operations on the mask
        with Image.open(
            os.path.join(self.mask_dir, mask_name), formats=["png"]
        ) as mask:
            out_mask = tv_tensors.Mask(self.transform_mask(mask))

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

        combined_video, out_mask = self.transform_together(combined_video, out_mask)
        assert len(combined_video.shape) == 4, (
            "Combined images must be of shape: (F, C, H, W) but is "
            + f"{combined_video.shape}"
        )

        out_video = concatenate_imgs(
            self.frames, self.select_frame_method, combined_video
        )
        out_residuals = out_video - torch.roll(out_video, -1, 0)

        return out_video, out_residuals, out_mask.squeeze().long(), img_name


def concat(
    img_list: Sequence[cvt.MatLike],
    indices: Sequence[int] | None,
    transform: Compose,
    loading_mode: LoadingMode,
) -> torch.Tensor:
    in_stack = np.stack(img_list, axis=0)
    if loading_mode == LoadingMode.GREYSCALE:
        in_stack = np.expand_dims(in_stack, -1)

    out_images = in_stack / [255.0]
    out_images = out_images.transpose(0, 3, 1, 2)  # F x C x H x W

    combined_imgs = torch.from_numpy(out_images)
    combined_imgs: torch.Tensor = transform(combined_imgs)

    if indices is not None:
        combined_imgs = combined_imgs[indices]

    return combined_imgs


def concatenate_imgs(
    frames: int,
    select_frame_method: Literal["consecutive", "specific"],
    imgs: torch.Tensor,
) -> torch.Tensor:
    """Concatenates the images based on the number of frames and the method of
    selecting frames.

    Args:
        frames: The number of frames to concatenate.
        select_frame_method: The method of selecting frames.
        img_list: The list of images to concatenate.
        transform: The transform to apply to the images.

    Returns:
        torch.Tensor: The concatenated images.

    Raises:
        ValueError: If the number of frames is not within [5, 10, 15, 20, 30].
        ValueError: If the method of selecting frames is not valid.
    """
    if frames == 30:
        return imgs

    CHOSEN_FRAMES_DICT = {
        5: [0, 6, 12, 18, 24],
        10: [0, 3, 6, 9, 12, 15, 18, 21, 24, 27],
        15: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        20: [
            0,
            2,
            4,
            6,
            8,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            22,
            24,
            26,
            28,
        ],
    }

    if frames < 30 and frames > 0:
        match select_frame_method:
            case "consecutive":
                indices = range(frames)
                return imgs[indices]
            case "specific":
                if frames not in CHOSEN_FRAMES_DICT:
                    raise ValueError(
                        f"Invalid number of frames ({frames}) for the specific "
                        + "frame selection method. "
                        + f"Ensure that it is within {sorted(CHOSEN_FRAMES_DICT.keys())}"
                    )
                return imgs[CHOSEN_FRAMES_DICT[frames]]

    raise ValueError(
        f"Invalid number of frames ({frames}), ensure that 0 < frames <= 30"
    )


def load_train_indices(
    dataset: (
        LGEDataset
        | CineDataset
        | TwoPlusOneDataset
        | TwoStreamDataset
        | ResidualTwoPlusOneDataset
    ),
    train_idxs_path: str,
    valid_idxs_path: str,
) -> tuple[list[int], list[int]]:
    """
    Loads the training and validation indices for the dataset.

    If the path to the indices are invalid, it then generates the indices in a
    possibly deterministic way. This method also sets the `dataset.train_idxs` and
    `dataset.valid_idxs` properties.

    Args:
        train_idxs_path: Path to training indices pickle file.
        valid_idxs_path: Path to validation indices pickle file.

    Returns:
        tuple[list[int], list[int]]: Training and Validation indices

    Raises:
        RuntimeError: If there are duplicates in the training and validation
        indices.
        RuntimeError: If patients have images in both the training and testing.
        AssertionError: If the training and validation indices are not disjoint.

    Example:
        lge_dataset = LGEDataset()
        load_train_indices(lge_dataset, train_idxs_path, valid_idxs_path)
    """
    if os.path.exists(train_idxs_path) and os.path.exists(valid_idxs_path):
        with open(train_idxs_path, "rb") as f:
            train_idxs: list[int] = pickle.load(f)
        with open(valid_idxs_path, "rb") as f:
            valid_idxs: list[int] = pickle.load(f)
        dataset.train_idxs = train_idxs
        dataset.valid_idxs = valid_idxs
        return train_idxs, valid_idxs

    names = os.listdir(dataset.img_dir)

    # Group patient files together so that all of a patient's files are in one group.
    # This is to ensure that all patient files are strictly only in training,
    # validation, or testing.
    grouped_names = {}
    blacklisted = [""]

    for i, name in enumerate(names):
        base = name.split("_")[0]

        if name not in blacklisted:
            if len(grouped_names) == 0:
                grouped_names[base] = [names[i]]
            else:
                if base in grouped_names:
                    grouped_names[base] += [names[i]]
                else:
                    grouped_names[base] = [names[i]]

    # Attach an index to each file
    for i in range(len(names)):
        tri = dataset[i]
        base = tri[2].split("_")[0]
        for x, name in enumerate(grouped_names[base]):
            if name == tri[2]:
                grouped_names[base][x] = [name, i]

    # Get indices for training, validation, and testing.
    length = len(dataset)
    val_len = length // 4

    train_idxs = []
    valid_idxs = []

    train_names: list[str] = []
    valid_names: list[str] = []

    train_len = length - val_len
    for patient in grouped_names:
        while len(train_idxs) < train_len:
            for i in range(len(grouped_names[patient])):
                name, idx = grouped_names[patient][i]
                train_idxs.append(idx)
                train_names.append(name)
            break

        else:
            while len(valid_idxs) < val_len:
                for i in range(len(grouped_names[patient])):
                    name, idx = grouped_names[patient][i]
                    valid_idxs.append(idx)
                    valid_names.append(name)
                break

    # Check to make sure no indices are repeated.
    for name in train_idxs:
        if name in valid_idxs:
            raise RuntimeError(f"Duplicate in train and valid indices exists: {name}")

    # Check to make sure no patients have images in both the training and testing.
    train_bases = {name.split("_")[0] for name in train_names}
    valid_bases = {name.split("_")[0] for name in valid_names}

    assert train_bases.isdisjoint(
        valid_bases
    ), "Patients have images in both the training and testing"

    dataset.train_idxs = train_idxs
    dataset.valid_idxs = valid_idxs

    with open(train_idxs_path, "wb") as f:
        pickle.dump(train_idxs, f)
    with open(valid_idxs_path, "wb") as f:
        pickle.dump(valid_idxs, f)

    return train_idxs, valid_idxs


def seed_worker(worker_id):
    """Sets the seed for the worker based on the initial seed.

    Args:
        worker_id: The worker ID (not used).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_trainval_data_subsets(
    train_dataset: (
        CineDataset
        | LGEDataset
        | TwoPlusOneDataset
        | TwoStreamDataset
        | ResidualTwoPlusOneDataset
    ),
    valid_dataset: (
        CineDataset
        | LGEDataset
        | TwoPlusOneDataset
        | TwoStreamDataset
        | ResidualTwoPlusOneDataset
        | None
    ) = None,
) -> tuple[Subset, Subset]:
    """Gets the subsets of the data as train/val splits from a superset consisting of
    both.

    Args:
        dataset: The original dataset.

    Returns:
        tuple[Subset, Subset]: Training and validation subsets.
    """
    if not valid_dataset:
        valid_dataset = train_dataset

    assert type(valid_dataset) is type(train_dataset), (
        "train and valid datasets are not of the same type! "
        + f"{type(train_dataset)} != {type(valid_dataset)}"
    )

    train_set = Subset(train_dataset, train_dataset.train_idxs)
    valid_set = Subset(valid_dataset, valid_dataset.valid_idxs)
    return train_set, valid_set


def get_trainval_dataloaders(
    dataset: LGEDataset | TwoPlusOneDataset,
) -> tuple[DataLoader, DataLoader]:
    """Gets the dataloaders of the data as train/val splits from a superset consisting
    of both.

    The dataloaders are created using the `SubsetRandomSampler` to ensure that the
    training and validation sets are disjoint. The dataloaders are also set to have a
    fixed seed for reproducibility.

    Args:
        dataset: The original dataset.

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation dataloaders.
    """
    # Define fixed seeds
    random.seed(SEED_CUS)
    torch.manual_seed(SEED_CUS)
    torch.cuda.manual_seed(SEED_CUS)
    torch.cuda.manual_seed_all(SEED_CUS)
    train_sampler = SubsetRandomSampler(dataset.train_idxs)

    torch.manual_seed(SEED_CUS)
    torch.cuda.manual_seed(SEED_CUS)
    torch.cuda.manual_seed_all(SEED_CUS)
    valid_sampler = SubsetRandomSampler(dataset.valid_idxs)

    g = torch.Generator()
    g.manual_seed(SEED_CUS)

    train_loader = DataLoader(
        dataset=dataset,
        sampler=train_sampler,
        num_workers=0,
        batch_size=dataset.batch_size,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    valid_loader = DataLoader(
        dataset=dataset,
        sampler=valid_sampler,
        num_workers=0,
        batch_size=dataset.batch_size,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, valid_loader


def get_class_weights(
    train_set: Subset,
) -> npt.NDArray[np.float32]:
    """Gets the class weights based on the occurrence of the classes in the training
    set.

    Args:
        train_set: The training set.

    Returns:
        npt.NDArray[np.float32]: The class weights.

    Raises:
        AssertionError: If the subset's dataset object has no `classification_mode`
        attribute.
        AssertionError: If the classification mode is not multilabel.
    """
    dataset = train_set.dataset

    assert any(
        isinstance(dataset, dataset_class)
        for dataset_class in [
            LGEDataset | CineDataset | TwoPlusOneDataset | TwoStreamDataset
        ]
    )

    assert (
        getattr(dataset, "classification_mode", None) is not None
    ), "Dataset has no attribute `classification_mode`"

    assert (
        dataset.classification_mode  # pyright: ignore[reportAttributeAccessIssue]
        == ClassificationMode.MULTILABEL_MODE
    )
    counts = np.array([0.0, 0.0, 0.0, 0.0])
    for _, masks, _ in [train_set[i] for i in range(len(train_set))]:
        class_occurrence = masks.sum(dim=(1, 2))
        counts = counts + class_occurrence.numpy()

    inv_counts = [1.0] / counts
    inv_counts = inv_counts / inv_counts.sum()

    return inv_counts
