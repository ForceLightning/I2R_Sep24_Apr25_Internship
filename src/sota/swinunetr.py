# -*- coding: utf-8 -*-
"""Swin U-NetR training script."""
from __future__ import annotations

# Standard Library
import logging
import sys
from typing import override

# PyTorch
import torch
from lightning.pytorch.cli import LightningArgumentParser

# First party imports
from cine import CineBaselineDataModule
from cli.common import CommonCLI
from models.sota.swinunetr.lightning_module import SwinUnetRLightningModule
from utils.logging import LOGGING_FORMAT

BATCH_SIZE_TRAIN = 2
torch.set_float32_matmul_precision("medium")
logger = logging.getLogger(__name__)


class SwinUnetRCLI(CommonCLI):
    """CLI class for SwinUnetR implementation."""

    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)

        defaults = self.default_arguments | {
            "image_loading_mode": "RGB",
            "dl_classification_mode": "MULTICLASS_MODE",
            "eval_classification_mode": "MULTICLASS_MODE",
            "trainer.max_epochs": 50,
            "model.classes": 4,
            "model.in_channels": 30,
        }

        parser.set_defaults(defaults)


if __name__ == "__main__":
    file_handler = logging.FileHandler(filename="logs/swinunetr.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, handlers=handlers)
    logger = logging.getLogger(__name__)

    cli = SwinUnetRCLI(
        SwinUnetRLightningModule,
        CineBaselineDataModule,
        save_config_callback=None,
        auto_configure_optimizers=False,
        parser_kwargs={
            "fit": {"default_config_files": ["./configs/swinunetr.yaml"]},
            "predict": {
                "default_config_files": [
                    "./configs/swinunetr.yaml",
                    "./configs/predict.yaml",
                ]
            },
        },
    )
