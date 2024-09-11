from __future__ import annotations

import os
import pickle
import random
from typing import Any, Literal, Sequence, override

import cv2
import numpy as np
import torch
from cv2 import typing as cvt
from numpy import typing as npt
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler
from torchvision.transforms import Compose

SEED_CUS = 1


class CineDataset(Dataset[Any]):
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
    ) -> None:
        super().__init__()
        # Set paths to CINE and mask directories
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        # Get a list of all the files in the CINE and mask directories
        self.img_list = os.listdir(self.img_dir)
        self.mask_list = os.listdir(self.mask_dir)

        # Init transforms
        self.transform_1 = transform_1
        self.transform_2 = transform_2

        # Define number of frames and how those frames are chosen (consecutive or
        # specific).
        self.frames = frames
        self.select_frame_method: Literal["consecutive", "specific"] = (
            select_frame_method
        )

        self.train_idxs: list[int]
        self.valid_idxs: list[int]

        self.batch_size = batch_size

        if mode != "test":
            self.load_train_indices(
                os.path.join(idxs_dir, "train_indices.pkl"),
                os.path.join(idxs_dir, "val_indices.pkl"),
            )

    def __len__(self) -> int:
        return len(self.img_list)

    @override
    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor | npt.NDArray[np.floating[Any]], str]:
        img_name = self.img_list[index]
        mask_name = self.img_list[index].split(".")[0] + ".nii.png"

        # Read the .tiff with 30 pages using cv2.imreadmulti instead of cv2.imread,
        # loaded as RBG.
        img_tuple = cv2.imreadmulti(
            os.path.join(self.img_dir, img_name), flags=cv2.IMREAD_COLOR
        )

        # cv2.imreadmulti returns a tuple of length 2, with the second value of the
        # tuple being the actual images.
        img_list = img_tuple[1]

        # Concatenate the images based on specific indices (subject to change).
        combined_imgs = CineDataset.concatenate_imgs(
            self.frames, self.select_frame_method, img_list, self.transform_1
        )

        # Perform necessary operations on the mask
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name))

        lab_mask = mask / [1.0]
        lab_mask = cv2.resize(lab_mask, (224, 224)).astype(np.float32)
        lab_mask = lab_mask[:, :, 0]  # H x W

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

        if self.transform_2:
            lab_mask_one_hot = self.transform_2(lab_mask_one_hot)

        return combined_imgs, lab_mask_one_hot, img_name

    @classmethod
    def concatenate_imgs(
        cls,
        frames: int,
        select_frame_method: Literal["consecutive", "specific"],
        img_list: Sequence[cvt.MatLike],
        transform: Compose,
    ) -> torch.Tensor:
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
            return CineDataset.concat(img_list, indices, transform)

        elif frames < 30 and frames > 0:
            if select_frame_method == "consecutive":
                indices = range(0, frames)
                return CineDataset.concat(img_list, indices, transform)
            elif select_frame_method == "specific":
                if frames in chosen_frames_dict:
                    indices = chosen_frames_dict[frames]
                    return CineDataset.concat(img_list, indices, transform)
                else:
                    raise ValueError(
                        "Invalid number of frames for the specific frame selection method. Ensure that it is within [5, 10, 15, 20, 30]"
                    )

        raise ValueError(
            "Invalid number of frames for the specific frame selection method. Ensure that it is within [5, 10, 15, 20, 30]"
        )

    @classmethod
    def concat(
        cls,
        img_list: Sequence[cvt.MatLike],
        indices: Sequence[int],
        transform: Compose,
    ) -> torch.Tensor:
        starting_idx = indices[0]
        first_img = img_list[starting_idx]
        tuned = first_img / [255.0]
        tuned = cv2.resize(tuned, (224, 224)).astype(np.float32)
        tuned = transform(tuned)

        combined_imgs = tuned.unsqueeze(0)

        for i in indices[1:]:
            img = img_list[i]
            lab_img = img / [255.0]
            lab_img = cv2.resize(lab_img, (224, 224)).astype(np.float32)
            lab_img = transform(lab_img)
            combined_imgs = torch.cat((combined_imgs, lab_img.unsqueeze(0)), 0)

        return combined_imgs

    def load_train_indices(
        self,
        train_idxs_path: str,
        valid_idxs_path: str,
    ) -> tuple[list[int], list[int]]:
        if os.path.exists(train_idxs_path) and os.path.exists(valid_idxs_path):
            with open(train_idxs_path, "rb") as f:
                train_idxs: list[int] = pickle.load(f)
            with open(valid_idxs_path, "rb") as f:
                valid_idxs: list[int] = pickle.load(f)
            self.train_idxs = train_idxs
            self.valid_idxs = valid_idxs
            return train_idxs, valid_idxs

        names = os.listdir(self.img_dir)

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
            tri = self[i]
            base = tri[2].split("_")[0]
            for x, name in enumerate(grouped_names[base]):
                if name == tri[2]:
                    grouped_names[base][x] = [name, i]

        # Get indices for training, validation, and testing.
        length = len(self)
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
                raise RuntimeError(
                    f"Duplicate in train and valid indices exists: {name}"
                )

        # Check to make sure no patients have images in both the training and testing.
        train_bases = set([name.split("_")[0] for name in train_names])
        valid_bases = set([name.split("_")[0] for name in valid_names])

        assert train_bases.isdisjoint(
            valid_bases
        ), "Patients have images in both the training and testing"

        self.train_idxs = train_idxs
        self.valid_idxs = valid_idxs

        if not os.path.exists(
            (train_indices_dir := os.path.dirname(os.path.normpath(train_idxs_path)))
        ):
            os.makedirs(train_indices_dir)
        if not os.path.exists(
            (valid_indices_dir := os.path.dirname(os.path.normpath(valid_idxs_path)))
        ):
            os.makedirs(valid_indices_dir)

        with open(train_idxs_path, "wb") as f:
            pickle.dump(train_idxs, f)
        with open(valid_idxs_path, "wb") as f:
            pickle.dump(valid_idxs, f)

        return train_idxs, valid_idxs


class LGEDataset(Dataset[Any]):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        idxs_dir: str,
        transform_1: Compose,
        transform_2: Compose,
        batch_size: int = 8,
        mode: Literal["train", "val", "test"] = "train",
    ) -> None:
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
            self.load_train_indices(
                os.path.join(idxs_dir, "train_indices.pkl"),
                os.path.join(idxs_dir, "val_indices.pkl"),
            )

    @override
    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor | npt.NDArray[np.floating[Any]], str]:
        img_name = self.img_list[index]
        mask_name = self.img_list[index].split(".")[0] + "_0000.nii.png"

        # Read the .tiff with 30 pages using cv2.imreadmulti instead of cv2.imread,
        # loaded as RBG.
        assert img_name.endswith(".png"), "Image not in .PNG format"

        img = cv2.imread(os.path.join(self.img_dir, img_name))
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name))

        lab_img = img / [255.0]
        lab_mask = mask / [1.0]

        lab_img = cv2.resize(lab_img, (224, 224))
        lab_mask = cv2.resize(lab_mask, (224, 224))

        lab_img = lab_img.astype(np.float32)
        lab_mask = lab_mask.astype(np.float32)[:, :, 0]

        lab_img = self.transform_1(lab_img)

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

        lab_mask_one_hot = self.transform_2(lab_mask_one_hot)

        return lab_img, lab_mask_one_hot, img_name

    def __len__(self) -> int:
        return len(self.img_list)

    def load_train_indices(
        self,
        train_idxs_path: str,
        valid_idxs_path: str,
    ) -> tuple[list[int], list[int]]:
        if os.path.exists(train_idxs_path) and os.path.exists(valid_idxs_path):
            with open(train_idxs_path, "rb") as f:
                train_idxs: list[int] = pickle.load(f)
            with open(valid_idxs_path, "rb") as f:
                valid_idxs: list[int] = pickle.load(f)
            self.train_idxs = train_idxs
            self.valid_idxs = valid_idxs
            return train_idxs, valid_idxs

        names = os.listdir(self.img_dir)

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
            tri = self[i]
            base = tri[2].split("_")[0]
            for x, name in enumerate(grouped_names[base]):
                if name == tri[2]:
                    grouped_names[base][x] = [name, i]

        # Get indices for training, validation, and testing.
        length = len(self)
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
                raise RuntimeError(
                    f"Duplicate in train and valid indices exists: {name}"
                )

        # Check to make sure no patients have images in both the training and testing.
        train_bases = set([name.split("_")[0] for name in train_names])
        valid_bases = set([name.split("_")[0] for name in valid_names])

        assert train_bases.isdisjoint(
            valid_bases
        ), "Patients have images in both the training and testing"

        self.train_idxs = train_idxs
        self.valid_idxs = valid_idxs

        return train_idxs, valid_idxs


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_trainval_data_subsets(
    dataset: LGEDataset | CineDataset,
) -> tuple[Subset, Subset]:
    train_set = Subset(dataset, dataset.train_idxs)
    valid_set = Subset(dataset, dataset.valid_idxs)
    return train_set, valid_set


def get_trainval_dataloaders(
    dataset: LGEDataset | CineDataset,
) -> tuple[DataLoader, DataLoader]:
    # Define fixed seeds
    random.seed(SEED_CUS)
    torch.manual_seed(SEED_CUS)
    torch.cuda.manual_seed(SEED_CUS)
    torch.cuda.manual_seed_all(SEED_CUS)
    a = list(range(len(dataset)))
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
