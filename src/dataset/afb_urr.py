# Standard Library
import os
from typing import Literal, Optional, override

# Image Libraries
import cv2
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE
from PIL import Image

# PyTorch
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose

# First party imports
from dataset.dataset import DefaultTransformsMixin, concatenate_imgs, load_train_indices
from utils.types import ClassificationMode, LoadingMode


class AFB_URRDataset(Dataset[tuple[Tensor, Tensor, str]], DefaultTransformsMixin):
    """AFB-URR compatible dataset."""

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        idxs_dir: str,
        frames: int,
        select_frame_method: Literal["consecutive", "specific", "specific + last"],
        transform_img: Compose,
        transform_mask: Compose,
        transform_together: Optional[Compose] = None,
        batch_size: int = 2,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
    ) -> None:
        """Initialise the AFB-URR compatible dataset.

        Args:
            img_dir: Path to the directory containing the cine images.
            mask_dir: Path to the directory containing the masks.
            idxs_dir: Path to the directory containing the indices.
            frames: Number of frames to select from the cine images.
            select_frame_method: Method to select frames.
            transform_img: Transforms to apply to the cine images.
            transform_mask: Transforms to apply to the masks.
            transform_together: Transforms to apply to both the cine images and masks.
            batch_size: Batch size.
            mode: Training/Inference mode.
            classification_mode: Classification mode.
            loading_mode: Loading mode.
            combine_train_val: Combine the train and validation datasets.

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
        self.select_frame_method: Literal[
            "consecutive", "specific", "specific + last"
        ] = select_frame_method
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
        """Return the length of the dataset."""
        return len(self.img_list)

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
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

        out_video, out_mask = self.transform_together(combined_video, out_mask)
        match self.select_frame_method:
            case "specific" | "consecutive":
                out_video = concatenate_imgs(
                    self.frames, self.select_frame_method, out_video
                )
            case _:
                out_video = self.select_frames_with_last(self.frames, out_video)

        f, c, h, w = out_video.shape
        out_video = out_video.reshape(f, c, h, w)

        return out_video, out_mask.squeeze().long(), img_name

    def select_frames_with_last(self, num_frames: int, imgs: Tensor) -> Tensor:
        """Select `num_frames` from imgs, including the last frame.

        Args:
            num_frames: Number of frames to use.
            imgs: Loaded cine images.

        Returns:
            Tensor: Tensor with selected frames only.

        """
        if num_frames == 30:
            return imgs

        assert (
            num_frames < 30 and num_frames > 0
        ), f"number of frames: {num_frames} must be <= 30 and > 0"

        idxs = list(reversed(range(29, -1, -30 // num_frames)))
        return imgs[idxs]
