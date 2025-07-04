# -*- coding: utf-8 -*-
"""PNS+ architecture training script."""
from __future__ import annotations

# Standard Library
import logging
import sys
from typing import override

# PyTorch
import torch
from lightning.pytorch.cli import LightningArgumentParser

# First party imports
from cli.common import CommonCLI
from models.sota.pns.lightning_module import PNSLightningModule
from two_plus_one import TwoPlusOneDataModule
from utils.logging import LOGGING_FORMAT

BATCH_SIZE_TRAIN = 2  # Default batch size for training.
NUM_FRAMES = 5  # Default number of frames.
torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)


class PNSPlusCLI(CommonCLI):
    """CLI class for PNS+ implementation."""

    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)

        defaults = self.default_arguments | {
            "image_loading_mode": "RGB",
            "dl_classification_mode": "MULTICLASS_MODE",
            "eval_classification_mode": "MULTILABEL_MODE",
            "trainer.max_epochs": 50,
        }

        parser.set_defaults(defaults)


if __name__ == "__main__":
    file_handler = logging.FileHandler(filename="logs/pns.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, handlers=handlers)
    logger = logging.getLogger(__name__)

    cli = PNSPlusCLI(
        PNSLightningModule,
        TwoPlusOneDataModule,
        save_config_callback=None,
        auto_configure_optimizers=False,
        parser_kwargs={
            "fit": {"default_config_files": ["./configs/pns.yaml"]},
            "predict": {
                "default_config_files": [
                    "./configs/pns.yaml",
                    "./configs/predict.yaml",
                ]
            },
        },
    )
