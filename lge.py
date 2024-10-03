# -*- coding: utf-8 -*-
"""LGE Baseline model training script.
"""
from __future__ import annotations

import os
import ssl
from typing import Union, override

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from torch.utils.data import DataLoader

from cine import LightningUnetWrapper
from dataset.dataset import LGEDataset, get_trainval_data_subsets
from utils import utils
from utils.prediction_writer import MaskImageWriter, get_output_dir_from_ckpt_path
from utils.utils import ClassificationMode, LoadingMode

BATCH_SIZE_TRAIN = 8  # Default batch size for training.
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.set_float32_matmul_precision("medium")
ssl._create_default_https_context = ssl._create_unverified_context


class LGEBaselineDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/train_val/",
        test_dir: str = "data/test/",
        indices_dir: str = "data/indices/",
        batch_size: int = BATCH_SIZE_TRAIN,
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        num_workers: int = 8,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
        augment: bool = False,
    ):
        """LGE MRI image data module.

        Args:
            data_dir: Path to the training and validation data.
            test_dir: Path to the test data.
            indices_dir: Path to the indices directory.
            batch_size: Batch size for training.
            classification_mode: Classification mode.
            num_workers: Number of workers for data loading.
            loading_mode: Image loading mode for the dataset.
            combine_train_val: Whether to combine train/val sets.
        """

        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.indices_dir = indices_dir
        self.batch_size = batch_size
        self.classification_mode = classification_mode
        self.num_workers = num_workers
        self.loading_mode = loading_mode
        self.combine_train_val = combine_train_val
        self.augment = augment

    @override
    def setup(self, stage):
        indices_dir = os.path.join(os.getcwd(), self.indices_dir)

        trainval_img_dir = os.path.join(os.getcwd(), self.data_dir, "LGE")
        trainval_mask_dir = os.path.join(os.getcwd(), self.data_dir, "masks")

        transforms_img, transforms_mask, transforms_together = utils.get_transforms(
            self.loading_mode, self.augment
        )

        trainval_dataset = LGEDataset(
            trainval_img_dir,
            trainval_mask_dir,
            indices_dir,
            transform_img=transforms_img,
            transform_mask=transforms_mask,
            transform_together=transforms_together,
            classification_mode=self.classification_mode,
            loading_mode=self.loading_mode,
            combine_train_val=self.combine_train_val,
        )
        assert len(trainval_dataset) > 0, "combined train/val set is empty"

        test_img_dir = os.path.join(os.getcwd(), self.test_dir, "LGE")
        test_mask_dir = os.path.join(os.getcwd(), self.test_dir, "masks")

        test_dataset = LGEDataset(
            test_img_dir,
            test_mask_dir,
            indices_dir,
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

            valid_dataset = LGEDataset(
                trainval_img_dir,
                trainval_mask_dir,
                indices_dir,
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

    def on_exception(self, exception):
        raise exception

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

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=False,
        )


class LGECLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        if self.subcommand is not None:
            if (config := self.config.get(self.subcommand)) is not None:
                if (version := config.get("version")) is not None:
                    name = utils.get_last_checkpoint_filename(version)
                    ModelCheckpoint.CHECKPOINT_NAME_LAST = (  # pyright: ignore[reportAttributeAccessIssue]
                        name
                    )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.add_lightning_class_args(
            ModelCheckpoint, "model_checkpoint_dice_weighted"
        )
        parser.add_argument("--version", type=Union[str, None], default=None)

        # Sets the checkpoint filename if version is provided.
        parser.link_arguments(
            "version",
            "model_checkpoint.filename",
            compute_fn=utils.get_checkpoint_filename,
        )
        parser.link_arguments(
            "version",
            "model_checkpoint_dice_weighted.filename",
            compute_fn=utils.get_best_weighted_avg_dice_filename,
        )
        parser.link_arguments("version", "trainer.logger.init_args.name")

        # Adds the classification mode argument
        parser.add_argument("--dl_classification_mode", type=str)
        parser.add_argument("--eval_classification_mode", type=str)
        parser.link_arguments(
            "dl_classification_mode",
            "model.dl_classification_mode",
            compute_fn=utils.get_classification_mode,
        )
        parser.link_arguments(
            "eval_classification_mode",
            "model.eval_classification_mode",
            compute_fn=utils.get_classification_mode,
        )
        parser.link_arguments(
            "dl_classification_mode",
            "data.classification_mode",
            compute_fn=utils.get_classification_mode,
        )

        parser.link_arguments("trainer.max_epochs", "model.total_epochs")

        # Sets the image color loading mode
        parser.add_argument("--image_loading_mode", type=Union[str, None], default=None)
        parser.link_arguments(
            "image_loading_mode", "data.loading_mode", compute_fn=utils.get_loading_mode
        )
        parser.link_arguments(
            "image_loading_mode",
            "model.loading_mode",
            compute_fn=utils.get_loading_mode,
        )

        # Link data.batch_size and model.batch_size
        parser.link_arguments(
            "data.batch_size", "model.batch_size", apply_on="instantiate"
        )

        # Prediction writer
        parser.add_lightning_class_args(MaskImageWriter, "prediction_writer")
        parser.link_arguments("image_loading_mode", "prediction_writer.loading_mode")
        parser.link_arguments(
            "model.weights_from_ckpt_path",
            "prediction_writer.output_dir",
            compute_fn=get_output_dir_from_ckpt_path,
        )

        parser.set_defaults(
            {
                "image_loading_mode": "RGB",
                "dl_classification_mode": "MULTICLASS_MODE",
                "eval_classification_mode": "MULTILABEL_MODE",
                "trainer.max_epochs": 50,
                "model.encoder_name": "resnet50",
                "model.encoder_weights": "imagenet",
                "model.in_channels": 3,  # 1 image, RGB channels.
                "model.classes": 4,
                "model_checkpoint.monitor": "loss/val",
                "model_checkpoint.save_last": True,
                "model_checkpoint.save_weights_only": True,
                "model_checkpoint.save_top_k": 1,
                "model_checkpoint.auto_insert_metric_name": False,
                "model_checkpoint_dice_weighted.monitor": "val/dice_weighted_avg",
                "model_checkpoint_dice_weighted.save_top_k": 1,
                "model_checkpoint_dice_weighted.save_weights_only": True,
                "model_checkpoint_dice_weighted.save_last": False,
                "model_checkpoint_dice_weighted.mode": "max",
                "model_checkpoint_dice_weighted.auto_insert_metric_name": False,
            }
        )


if __name__ == "__main__":
    cli = LGECLI(
        LightningUnetWrapper,
        LGEBaselineDataModule,
        save_config_callback=None,
        auto_configure_optimizers=False,
        parser_kwargs={
            "fit": {"default_config_files": ["./configs/lge.yaml"]},
            "predict": {
                "default_config_files": [
                    "./configs/lge.yaml",
                    "./configs/predict.yaml",
                ]
            },
        },
    )
