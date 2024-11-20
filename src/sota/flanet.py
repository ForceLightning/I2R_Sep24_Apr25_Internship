# -*- coding: utf-8 -*-
"""Two-plus-one architecture training script."""
from __future__ import annotations

# Standard Library
import os
from typing import Literal, override

# PyTorch
import lightning as L
import torch
from lightning.pytorch.cli import LightningArgumentParser
from torch.nn.common_types import _size_2_t
from torch.utils.data import DataLoader

# First party imports
from cli.common import I2RInternshipCommonCLI
from dataset.dataset import get_trainval_data_subsets
from dataset.flanet import FLANetDataset
from models.sota.fla_net.lightning_module import FLANetLightningModule
from utils.types import ClassificationMode, LoadingMode

BATCH_SIZE_TRAIN = 2  # Default batch size for training.
NUM_FRAMES = 5  # Default number of frames.
torch.set_float32_matmul_precision("medium")


class FLANetDataModule(L.LightningDataModule):
    """DataModule for the FLA-Net implementation."""

    def __init__(
        self,
        data_dir: str = "data/train_val/",
        test_dir: str = "data/test/",
        indices_dir: str = "data/indices/",
        batch_size: int = BATCH_SIZE_TRAIN,
        image_size: _size_2_t = (224, 224),
        frames: int = 30,
        select_frame_method: Literal["consecutive", "specific"] = "specific",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        num_workers: int = 8,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
        augment: bool = False,
        dummy_predict: bool = False,
    ):
        """Initialise the FLA-Net DataModule.

        Args:
            data_dir: Path to the training and validation data.
            test_dir: Path to the test data.
            indices_dir: Path to the indices directory.
            batch_size: Batch size for the DataLoader.
            image_size: Size of the image.
            frames: Number of frames.
            select_frame_method: Method to select frames.
            classification_mode: Classification mode.
            num_workers: Number of workers for the DataLoader.
            loading_mode: Loading mode.
            combine_train_val: Combine training and validation data.
            augment: Whether to augment the data.
            dummy_predict: Use train/val in predict dataloaders.

        """
        super().__init__()
        self.save_hyperparameters()
        self.frames = frames
        self.select_frame_method: Literal["consecutive", "specific"] = (
            select_frame_method
        )
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.indices_dir = indices_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.classification_mode = classification_mode
        self.num_workers = num_workers
        self.loading_mode = loading_mode
        self.combine_train_val = combine_train_val
        self.augment = augment
        self.dummy_predict = dummy_predict

    @override
    def setup(self, stage):
        indices_dir = os.path.join(os.getcwd(), self.indices_dir)

        trainval_img_dir = os.path.join(os.getcwd(), self.data_dir, "Cine")
        trainval_mask_dir = os.path.join(os.getcwd(), self.data_dir, "masks")

        transforms_img, transforms_mask, transforms_together = (
            FLANetDataset.get_default_transforms(
                self.loading_mode, self.augment, self.image_size
            )
        )

        trainval_dataset = FLANetDataset(
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
            image_size=self.image_size,
        )

        assert len(trainval_dataset) > 0, "combined train/val set is empty"

        test_img_dir = os.path.join(os.getcwd(), self.test_dir, "Cine")
        test_mask_dir = os.path.join(os.getcwd(), self.test_dir, "masks")

        test_dataset = FLANetDataset(
            test_img_dir,
            test_mask_dir,
            indices_dir,
            frames=self.frames,
            select_frame_method=self.select_frame_method,
            transform_img=transforms_img,
            transform_mask=transforms_mask,
            transform_together=transforms_together,
            classification_mode=self.classification_mode,
            loading_mode=self.loading_mode,
            combine_train_val=self.combine_train_val,
            image_size=self.image_size,
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

            valid_dataset = FLANetDataset(
                test_img_dir,
                test_mask_dir,
                indices_dir,
                frames=self.frames,
                select_frame_method=self.select_frame_method,
                transform_img=transforms_img,
                transform_mask=transforms_mask,
                transform_together=transforms_together,
                classification_mode=self.classification_mode,
                loading_mode=self.loading_mode,
                combine_train_val=self.combine_train_val,
                image_size=self.image_size,
            )

            train_set, valid_set = get_trainval_data_subsets(
                trainval_dataset, valid_dataset
            )

            self.train = train_set
            self.val = valid_set
            self.test = test_dataset

    def train_dataloader(self):
        """Get the training dataloader."""
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=True,
        )

    def val_dataloader(self):
        """Get the validation dataloader."""
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        """Get the test dataloader."""
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    @override
    def predict_dataloader(self):
        test_loader = DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=False,
        )

        if self.dummy_predict:
            train_loader = DataLoader(
                self.train,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=self.num_workers,
                drop_last=True,
                persistent_workers=True if self.num_workers > 0 else False,
                shuffle=False,
            )
            if not self.combine_train_val:
                valid_loader = DataLoader(
                    self.train,
                    batch_size=self.batch_size,
                    pin_memory=True,
                    num_workers=self.num_workers,
                    drop_last=True,
                    persistent_workers=True if self.num_workers > 0 else False,
                    shuffle=False,
                )

                return (train_loader, valid_loader, test_loader)
            return (train_loader, test_loader)
        return test_loader


class FLANetCLI(I2RInternshipCommonCLI):
    """CLI class for FLA-Net SOTA implementation."""

    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)

        defaults = self.default_arguments | {
            "image_loading_mode": "RGB",
            "dl_classification_mode": "MULTICLASS_MODE",
            "eval_classification_mode": "MULTILABEL_MODE",
            "trainer.max_epochs": 50,
            "model_architecture": "UNET",
            "model.encoder_name": "timm-res2net50_26w_4s",
            "model.encoder_weights": "imagenet",
            "model.in_channels": 3,
            "model.classes": 4,
        }

        parser.set_defaults(defaults)


if __name__ == "__main__":
    cli = FLANetCLI(
        FLANetLightningModule,
        FLANetDataModule,
        save_config_callback=None,
        auto_configure_optimizers=False,
        parser_kwargs={
            "fit": {"default_config_files": ["./configs/flanet.yaml"]},
            "predict": {
                "default_config_files": [
                    "./configs/flanet.yaml",
                    "./configs/predict.yaml",
                ]
            },
        },
    )
