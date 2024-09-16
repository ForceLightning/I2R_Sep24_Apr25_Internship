from __future__ import annotations

import os
import pickle
import random
from typing import Literal, Sequence, override

import cv2
import numpy as np
import torch
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE
from cv2 import typing as cvt
from numpy import typing as npt
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler
from torchvision.transforms import Compose

from utils.utils import ClassificationMode, LoadingMode

SEED_CUS = 1  # RNG seed.


class LGEDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        idxs_dir: str,
        transform_1: Compose,
        transform_2: Compose,
        batch_size: int = 8,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
    ) -> None:
        """LGE dataset for the cardiac LGE MRI images.

        Args:
            img_dir: The directory containing the LGE images.
            mask_dir: The directory containing the masks for the LGE images.
            idxs_dir: The directory containing the indices for the training and
            validation sets.
            transform_1: The transform to apply to the images.
            transform_2: The transform to apply to the masks.
            batch_size: The batch size for the dataset.
            mode: The mode of the dataset.
            classification_mode: The classification mode for the dataset.
            loading_mode: Determines the cv2.imread flags for the images.

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

        self.transform_1 = transform_1
        self.transform_2 = transform_2

        self.train_idxs: list[int]
        self.valid_idxs: list[int]

        self.batch_size = batch_size
        if mode != "test":
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

        # Read the .tiff with 30 pages using cv2.imreadmulti instead of cv2.imread,
        # loaded as RBG.
        assert img_name.endswith(".png"), "Image not in .PNG format"

        img = cv2.imread(os.path.join(self.img_dir, img_name), flags=self._imread_mode)
        mask = cv2.imread(
            os.path.join(self.mask_dir, mask_name), flags=self._imread_mode
        )

        lab_img = img / [255.0]
        lab_mask = mask / [1.0]

        lab_img = lab_img.astype(np.float32)
        lab_mask = lab_mask.astype(np.float32)[:, :, 0]

        out_img: torch.Tensor = self.transform_1(lab_img)
        out_mask: torch.Tensor

        if self.classification_mode == ClassificationMode.MULTILABEL_MODE:
            # NOTE: This turns the problem into a multilabel segmentation problem.
            # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
            # bitwise or operations to adhere to those conditions.
            lab_mask_one_hot = F.one_hot(
                torch.from_numpy(lab_mask).long(), num_classes=4
            )  # H x W x C
            lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
                lab_mask_one_hot[:, :, 3]
            )
            lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
                lab_mask_one_hot[:, :, 2]
            )
            lab_mask_one_hot = lab_mask_one_hot.bool().permute(-1, 0, 1)

            out_mask = self.transform_2(lab_mask_one_hot)
        elif self.classification_mode == ClassificationMode.MULTICLASS_MODE:
            out_mask = self.transform_2(lab_mask)
        else:
            raise NotImplementedError(
                f"The mode {self.classification_mode.name} is not implemented."
            )
        return out_img, out_mask, img_name

    def __len__(self) -> int:
        return len(self.img_list)


class CineDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        idxs_dir: str,
        transform_1: Compose,
        transform_2: Compose,
        batch_size: int = 4,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
    ) -> None:
        """Dataset for the Cine baseline implementation

        Args:
            img_dir: Path to the directory containing the images.
            mask_dir: Path to the directory containing the masks.
            idxs_dir: Path to the directory containing the indices.
            transform_1: Transform to apply to the images.
            transform_2: Transform to apply to the masks.
            batch_size: Batch size for the dataset.
            mode: Runtime mode.
            classification_mode: Classification mode for the dataset.
            loading_mode: Determines the cv2.imread flags for the images.

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

        self.transform_1 = transform_1
        self.transform_2 = transform_2

        self.train_idxs: list[int]
        self.valid_idxs: list[int]

        self.batch_size = batch_size
        self.mode = mode
        self.classification_mode = classification_mode
        self.loading_mode = loading_mode
        self._imread_mode = (
            IMREAD_COLOR if self.loading_mode == LoadingMode.RGB else IMREAD_GRAYSCALE
        )

        if mode != "test":
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

        _, img_list = cv2.imreadmulti(
            os.path.join(self.img_dir, img_name), flags=self._imread_mode
        )

        combined_imgs = concatenate_imgs(
            frames=30,
            select_frame_method="consecutive",
            img_list=img_list,
            transform=self.transform_1,
        )

        f, c, h, w = combined_imgs.shape
        combined_imgs = combined_imgs.reshape(f * c, h, w)

        mask = cv2.imread(os.path.join(self.mask_dir, mask_name))
        lab_mask = mask / [1.0]
        lab_mask = lab_mask.astype(np.float32)
        lab_mask = lab_mask[:, :, 0]  # H x W
        out_mask: torch.Tensor

        if self.classification_mode == ClassificationMode.MULTILABEL_MODE:
            # NOTE: This turns the problem into a multilabel segmentation problem.
            # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
            # bitwise or operations to adhere to those conditions.
            lab_mask_one_hot = F.one_hot(
                torch.from_numpy(lab_mask).long(), num_classes=4
            )  # H x W x C
            lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
                lab_mask_one_hot[:, :, 3]
            )
            lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
                lab_mask_one_hot[:, :, 2]
            )

            lab_mask_one_hot = lab_mask_one_hot.bool().permute(-1, 0, 1)

            out_mask = self.transform_2(lab_mask_one_hot)

        elif self.classification_mode == ClassificationMode.MULTICLASS_MODE:
            out_mask = self.transform_2(lab_mask)
        else:
            raise NotImplementedError(
                f"The mode {self.classification_mode.name} is not implemented"
            )

        return combined_imgs, out_mask, img_name

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
        transform_1: Compose,
        transform_2: Compose,
        batch_size: int = 2,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
    ) -> None:
        """Cine dataset for the cardiac cine MRI images.

        Args:
            img_dir: The directory containing the CINE images.
            mask_dir: The directory containing the masks for the CINE images.
            idxs_dir: The directory containing the indices for the training and
                validation sets.
            frames: The number of frames to concatenate.
            select_frame_method: The method of selecting frames to concatenate.
            transform_1: The transform to apply to the images.
            transform_2: The transform to apply to the masks.
            batch_size: The batch size for the dataset.
            mode: The mode of the dataset.
            classification_mode: The classification mode for the dataset.
            loading_mode: Determines the cv2.imread flags for the images.

        Raises:
            NotImplementedError: If the classification mode is not implemented.
            RuntimeError: If the indices fail to load.
            AssertionError: If the indices are not disjoint.
        """
        super().__init__(
            img_dir,
            mask_dir,
            idxs_dir,
            transform_1,
            transform_2,
            batch_size,
            mode,
            classification_mode,
        )
        self.frames = frames
        self.select_frame_method: Literal["consecutive", "specific"] = (
            select_frame_method
        )

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        img_name = self.img_list[index]
        mask_name = self.img_list[index].split(".")[0] + ".nii.png"

        # Read the .tiff with 30 pages using cv2.imreadmulti instead of cv2.imread,
        # loaded as RBG.
        # XXX: Check if it actually is RBG rather than RGB.
        _, img_list = cv2.imreadmulti(
            os.path.join(self.img_dir, img_name), flags=self._imread_mode
        )

        # Concatenate the images based on specific indices (subject to change).
        combined_imgs = concatenate_imgs(
            self.frames, self.select_frame_method, img_list, self.transform_1
        )

        # Perform necessary operations on the mask
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name))

        lab_mask = mask / [1.0]
        lab_mask = lab_mask[:, :, 0]  # H x W

        if self.classification_mode == ClassificationMode.MULTILABEL_MODE:
            # NOTE: This turns the problem into a multilabel segmentation problem.
            # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
            # bitwise or operations to adhere to those conditions.
            lab_mask_one_hot = F.one_hot(
                torch.from_numpy(lab_mask).long(), num_classes=4
            )  # H x W x C
            lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
                lab_mask_one_hot[:, :, 3]
            )
            lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
                lab_mask_one_hot[:, :, 2]
            )

            lab_mask_one_hot = lab_mask_one_hot.bool().permute(-1, 0, 1)

            lab_mask_out: torch.Tensor = self.transform_2(lab_mask_one_hot)

        elif self.classification_mode == ClassificationMode.MULTICLASS_MODE:
            lab_mask_out: torch.Tensor = self.transform_2(lab_mask)
        else:
            raise NotImplementedError(
                f"The mode {self.classification_mode.name} is not implemented"
            )

        return combined_imgs, lab_mask_out, img_name


class TwoStreamDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        lge_dir: str,
        cine_dir: str,
        mask_dir: str,
        idxs_dir: str,
        transform_1: Compose,
        transform_2: Compose,
        batch_size: int = 8,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
    ):
        """Two stream dataset for the cardiac LGE MRI images.

        Args:
            lge_dir: The directory containing the LGE images.
            cine_dir: The directory containing the CINE images.
            mask_dir: The directory containing the masks for the LGE images.
            idxs_dir: The directory containing the indices for the training and
            validation sets.
            transform_1: The transform to apply to the images.
            transform_2: The transform to apply to the masks.
            batch_size: The batch size for the dataset.
            mode: The mode of the dataset.
            classification_mode: The classification mode for the dataset.
            loading_mode: Determines the cv2.imread flags for the images.

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

        self.transform_1 = transform_1
        self.transform_2 = transform_2

        self.batch_size = batch_size
        self.train_idxs: list[int]
        self.valid_idxs: list[int]

        if mode != "test":
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
        return self.cine_dir

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
        mask_name = self.mask_list[index].split(".")[0] + "_0000.nii.png"
        cine_name = self.cine_list[index].split(".")[0] + "_0000.nii.tiff"

        if not lge_name.endswith(".png"):
            raise ValueError("Invalid image type for file: {lge_name}")

        # PERF: See if these can be turned into PIL operations (to maintain RGB)

        lge = cv2.imread(os.path.join(self.lge_dir, lge_name), self._imread_mode)
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), self._imread_mode)
        _, cine_list = cv2.imreadmulti(
            os.path.join(self.cine_dir, cine_name), flags=self._imread_mode
        )

        # Concatenate the images based on specific indices (subject to change).
        combined_cines = concatenate_imgs(
            frames=30,
            select_frame_method="consecutive",
            img_list=cine_list,
            transform=self.transform_1,
        )

        # Perform transformations on mask
        lab_mask = mask / [1.0]
        lab_mask = lab_mask[:, :, 0]
        if self.classification_mode == ClassificationMode.MULTILABEL_MODE:
            # NOTE: This turns the problem into a multilabel segmentation problem.
            # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
            # bitwise or operations to adhere to those conditions.
            lab_mask_one_hot = F.one_hot(
                torch.from_numpy(lab_mask).long(), num_classes=4
            )  # H x W x C
            lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
                lab_mask_one_hot[:, :, 3]
            )
            lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
                lab_mask_one_hot[:, :, 2]
            )

            lab_mask_one_hot = lab_mask_one_hot.bool().permute(-1, 0, 1)

            lab_mask_out: torch.Tensor = self.transform_2(lab_mask_one_hot)
        elif self.classification_mode == ClassificationMode.MULTICLASS_MODE:
            lab_mask_out: torch.Tensor = self.transform_2(lab_mask)
        else:
            raise NotImplementedError(
                f"The mode {self.classification_mode.name} is not implemented"
            )

        # Perform transformations on LGE
        lge = lge / [255.0]
        lge_out: torch.Tensor = self.transform_1(lge)

        return lge_out, combined_cines, lab_mask_out, lge_name

    def __len__(self):
        return len(self.cine_list)


def concat(
    img_list: Sequence[cvt.MatLike],
    indices: Sequence[int] | None,
    transform: Compose,
) -> torch.Tensor:
    in_stack = np.stack(img_list, axis=0)
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
    img_list: Sequence[cvt.MatLike],
    transform: Compose,
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
    chosen_frames_dict = {
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

    if frames == 30:
        indices = range(0, 30)
        return concat(img_list, indices, transform)

    elif frames < 30 and frames > 0:
        if select_frame_method == "consecutive":
            indices = range(0, frames)
            return concat(img_list, indices, transform)
        elif select_frame_method == "specific":
            if frames in chosen_frames_dict:
                indices = chosen_frames_dict[frames]
                return concat(img_list, indices, transform)
            else:
                raise ValueError(
                    "Invalid number of frames for the specific frame selection method. Ensure that it is within [5, 10, 15, 20, 30]"
                )

    raise ValueError(
        "Invalid number of frames for the specific frame selection method. Ensure that it is within [5, 10, 15, 20, 30]"
    )


def load_train_indices(
    dataset: LGEDataset | CineDataset | TwoPlusOneDataset | TwoStreamDataset,
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
                train_names.append(idx)
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
    train_bases = set([name.split("_")[0] for name in train_names])
    valid_bases = set([name.split("_")[0] for name in valid_names])

    assert train_bases.isdisjoint(
        valid_bases
    ), "Patients have images in both the training and testing"

    dataset.train_idxs = train_idxs
    dataset.valid_idxs = valid_idxs

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
    dataset: CineDataset | LGEDataset | TwoPlusOneDataset,
) -> tuple[Subset, Subset]:
    """Gets the subsets of the data as train/val splits from a superset consisting of
    both.

    Args:
        dataset: The original dataset.

    Returns:
        tuple[Subset, Subset]: Training and validation subsets.
    """
    train_set = Subset(dataset, dataset.train_idxs)
    valid_set = Subset(dataset, dataset.valid_idxs)
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

    assert (
        getattr(dataset, "classification_mode", None) is not None
    ), f"Dataset has no attribute `classification_mode`"

    assert dataset.classification_mode == ClassificationMode.MULTILABEL_MODE
    counts = np.array([0.0, 0.0, 0.0, 0.0])
    for _, masks, _ in [train_set[i] for i in range(len(train_set))]:
        class_occurrence = masks.sum(dim=(1, 2))
        counts = counts + class_occurrence.numpy()

    inv_counts = [1.0] / counts
    inv_counts = inv_counts / inv_counts.sum()

    return inv_counts
