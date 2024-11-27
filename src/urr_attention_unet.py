# -*- coding: utf-8 -*-
"""Attention-based U-Net on residual frame information with uncertainty."""

from __future__ import annotations

# PyTorch
import torch

# First party imports
from attention_unet import ResidualAttentionCLI, ResidualTwoPlusOneDataModule
from models.attention.urr.lightning_module import URRResidualAttentionLightningModule

BATCH_SIZE_TRAIN = 2  # Default batch size for training.
NUM_FRAMES = 5  # Default number of frames.
torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
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
