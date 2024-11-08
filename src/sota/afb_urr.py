# -*- coding: utf-8 -*-
"""AFB-URR architecture training script."""
from __future__ import annotations

# Standard Library
import os
from typing import Literal, override

# PyTorch
import lightning as L
import torch
from lightning.pytorch.cli import LightningArgumentParser
from torch.utils.data import DataLoader

# First party imports
from cli.common import CommonCLI
from dataset.afb_urr import AFB_URRDataset
from dataset.dataset import get_trainval_data_subsets
from models.sota.afb_urr.lightning_module import AFB_URRLightningModule
from utils.types import ClassificationMode, LoadingMode

BATCH_SIZE_TRAIN = 2  # Default batch size for training.
NUM_FRAMES = 5  # Default number of frames.
torch.set_float32_matmul_precision("medium")


class AFB_URRDataModule(L.LightningDataModule):
    """AFB-URR DataModule wrapper for compatible dataset."""

    def __init__(
        self,
        data_dir: str = "data/train_val/",
        test_dir: str = "data/test/",
        indices_dir: str = "data/indices/",
        batch_size: int = BATCH_SIZE_TRAIN,
        frames: int = NUM_FRAMES,
        select_frame_method: Literal[
            "consecutive", "specific", "specific + last"
        ] = "specific + last",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        num_workers: int = 8,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
        augment: bool = False,
    ) -> None:
        """Initialise the AFB-URR DataModule.

        Args:
           data_dir: Path to data directory.
           test_dir: Path to test data directory.
           indices_dir: Path to train/val split indices directory.
           batch_size: Batch size for dataloader.
           frames: Number of frames to use.
           select_frame_method: How to select frames if `frames` < 30.
           classification_mode: How to load masks.
           num_workers: Number of dataloader workers.
           loading_mode: Image loading mode (RGB/GREYSCALE).
           combine_train_val: Whether to combine train/val.
           augment: Whether to augment data.

        """
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.indices_dir = indices_dir
        self.batch_size = batch_size
        self.frames = frames
        self.select_frame_method: Literal[
            "consecutive", "specific", "specific + last"
        ] = select_frame_method
        self.classification_mode = classification_mode
        self.num_workers = num_workers
        self.loading_mode = loading_mode
        self.combine_train_val = combine_train_val
        self.augment = augment

    @override
    def setup(self, stage):
        indices_dir = os.path.join(os.getcwd(), self.indices_dir)

        trainval_img_dir = os.path.join(os.getcwd(), self.data_dir, "Cine")
        trainval_mask_dir = os.path.join(os.getcwd(), self.data_dir, "masks")

        # Get transforms for the CINE images, masks, and combined transforms.
        transforms_img, transforms_mask, transforms_together = (
            AFB_URRDataset.get_default_transforms(self.loading_mode, self.augment)
        )

        trainval_dataset = AFB_URRDataset(
            trainval_img_dir,
            trainval_mask_dir,
            indices_dir,
            frames=self.frames,
            select_frame_method=self.select_frame_method,
            transform_img=transforms_img,
            transform_mask=transforms_mask,
            transform_together=transforms_together,
            classification_mode=self.classification_mode,
            loading_mode=self.loading_mode,
            combine_train_val=self.combine_train_val,
        )
        assert len(trainval_dataset) > 0, "combined train/val set is empty"

        test_img_dir = os.path.join(os.getcwd(), self.test_dir, "Cine")
        test_mask_dir = os.path.join(os.getcwd(), self.test_dir, "masks")

        test_dataset = AFB_URRDataset(
            test_img_dir,
            test_mask_dir,
            indices_dir,
            frames=self.frames,
            select_frame_method=self.select_frame_method,
            transform_img=transforms_img,
            transform_mask=transforms_mask,
            mode="test",
            classification_mode=self.classification_mode,
            loading_mode=self.loading_mode,
            combine_train_val=self.combine_train_val,
        )

        if self.combine_train_val:
            self.train = trainval_dataset
            self.val = test_dataset
            self.test = test_dataset
        else:
            assert (idx := max(trainval_dataset.train_idxs)) < len(
                trainval_dataset
            ), f"Malformed training indices: {idx} for dataset of len: {len(trainval_dataset)}"

            assert (idx := max(trainval_dataset.valid_idxs)) < len(
                trainval_dataset
            ), f"Malformed training indices: {idx} for dataset of len: {len(trainval_dataset)}"

            valid_dataset = AFB_URRDataset(
                trainval_img_dir,
                trainval_mask_dir,
                indices_dir,
                frames=self.frames,
                select_frame_method=self.select_frame_method,
                transform_img=transforms_img,
                transform_mask=transforms_mask,
                classification_mode=self.classification_mode,
                loading_mode=self.loading_mode,
                combine_train_val=self.combine_train_val,
            )

            train_set, valid_set = get_trainval_data_subsets(
                trainval_dataset, valid_dataset
            )

            self.train = train_set
            self.val = valid_set
            self.test = test_dataset

    @override
    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=True,
        )

    @override
    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    @override
    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    @override
    def predict_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=False,
        )


class AFB_URR_CLI(CommonCLI):
    """CLI class for AFB-URR Implementation."""

    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)

        defaults = self.default_arguments | {
            "model.budget": 300000,
            "model.classes": 4,
            "model.in_channels": 3,
            "image_loading_mode": "RGB",
            "dl_classification_mode": "MULTICLASS_MODE",
            "eval_classification_mode": "MULTILABEL_MODE",
            "trainer.max_epochs": 50,
        }

        parser.set_defaults(defaults)


if __name__ == "__main__":
    cli = AFB_URR_CLI(
        AFB_URRLightningModule,
        AFB_URRDataModule,
        save_config_callback=None,
        auto_configure_optimizers=False,
        parser_kwargs={
            "fit": {"default_config_files": ["./configs/afb_urr.yaml"]},
            "predict": {
                "default_config_files": [
                    "./configs/afb_urr.yaml",
                    "./configs/predict.yaml",
                ]
            },
        },
    )
