from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Literal, override

import cv2
import lightning as L
import numpy as np
import segmentation_models_pytorch as smp
import torch
from cine import LightningUnetWrapper
from cv2 import typing as cvt
from dataset.dataset import LGEDataset, get_trainval_data_subsets
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
    LRSchedulerCallable,
    OptimizerCallable,
)
from numpy import typing as npt
from segmentation_models_pytorch.losses.dice import DiceLoss
from torch import nn
from torch.nn import functional as F
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.segmentation import GeneralizedDiceScore
from torchvision.transforms import v2
from torchvision.transforms.transforms import Compose
from two_plus_one import LightningGradualWarmupScheduler

BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VAL = 8
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
LEARNING_RATE = 1e-4
NUM_FRAMES = 5
SEED_CUS = 1  # RNG seed.
torch.set_float32_matmul_precision("medium")


class LGEBaselineDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/train_val-20240905T025601Z-001/train_val/",
        test_dir: str = "data/test-20240905T012341Z-001/test/",
        indices_dir: str = "data/indices/",
        batch_size: int = BATCH_SIZE_TRAIN,
    ):

        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.indices_dir = indices_dir
        self.batch_size = batch_size

    @override
    def setup(self, stage):
        indices_dir = os.path.join(os.getcwd(), self.indices_dir)

        trainval_img_dir = os.path.join(os.getcwd(), self.data_dir, "LGE")
        trainval_mask_dir = os.path.join(os.getcwd(), self.data_dir, "masks")
        transforms_img = Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        transforms_mask = Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        trainval_dataset = LGEDataset(
            trainval_img_dir,
            trainval_mask_dir,
            indices_dir,
            transform_1=transforms_img,
            transform_2=transforms_mask,
        )
        assert len(trainval_dataset) > 0, "combined train/val set is empty"

        assert (idx := max(trainval_dataset.train_idxs)) < len(
            trainval_dataset
        ), f"Malformed training indices: {idx} for dataset of len: {len(trainval_dataset)}"

        assert (idx := max(trainval_dataset.valid_idxs)) < len(
            trainval_dataset
        ), f"Malformed training indices: {idx} for dataset of len: {len(trainval_dataset)}"

        train_set, valid_set = get_trainval_data_subsets(trainval_dataset)

        test_img_dir = os.path.join(os.getcwd(), self.test_dir, "LGE")
        test_mask_dir = os.path.join(os.getcwd(), self.test_dir, "masks")

        test_dataset = LGEDataset(
            test_img_dir,
            test_mask_dir,
            indices_dir,
            transform_1=transforms_img,
            transform_2=transforms_mask,
            mode="test",
        )

        self.train = train_set
        self.val = valid_set
        self.test = test_dataset

    def on_exception(self, exception):
        raise exception

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8,
            drop_last=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8,
            persistent_workers=True,
        )


class CineCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(AdamW)
        parser.add_lr_scheduler_args(LightningGradualWarmupScheduler)
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.set_defaults(
            {
                "trainer.max_epochs": 50,
                "trainer.logger": {
                    "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                    "init_args": {
                        "save_dir": os.path.join(
                            os.getcwd(), "checkpoints/lge-baseline"
                        )
                    },
                },
                "model.encoder_name": "resnet50",
                "model.encoder_weights": "imagenet",
                "model.in_channels": 3,
                "model.classes": 4,
                "model_checkpoint.monitor": "val_loss",
                "model_checkpoint.dirpath": os.path.join(
                    os.getcwd(), "checkpoints/lge-baseline"
                ),
                "model_checkpoint.save_weights_only": True,
                "model_checkpoint.save_top_k": 1,
            }
        )


if __name__ == "__main__":
    cli = CineCLI(
        LightningUnetWrapper,
        LGEBaselineDataModule,
        # save_config_kwargs={
        #     "config_filename": "config.yaml",
        #     "multifile": True,
        # },
        save_config_callback=None,
    )
