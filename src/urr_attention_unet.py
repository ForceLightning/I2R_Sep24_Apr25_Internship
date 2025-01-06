# -*- coding: utf-8 -*-
"""Attention-based U-Net on residual frame information with uncertainty."""

from __future__ import annotations

# Standard Library
import logging

# PyTorch
import torch

# First party imports
from attention_unet import ResidualAttentionCLI, ResidualTwoPlusOneDataModule
from models.attention.urr import URRResidualAttentionLightningModule
from utils.logging import LOGGING_FORMAT

BATCH_SIZE_TRAIN = 2  # Default batch size for training.
NUM_FRAMES = 5  # Default number of frames.
torch.set_float32_matmul_precision("medium")

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        filename="logs/urr_attention_unet.log",
        format=LOGGING_FORMAT,
    )
    cli = ResidualAttentionCLI(
        URRResidualAttentionLightningModule,
        ResidualTwoPlusOneDataModule,
        save_config_callback=None,
        auto_configure_optimizers=False,
        parser_kwargs={
            "fit": {"default_config_files": ["./configs/residual_attention.yaml"]},
            "predict": {
                "default_config_files": [
                    "./configs/residual_attention.yaml",
                    "./configs/predict.yaml",
                ]
            },
        },
    )
