# -*- coding: utf-8 -*-
"""Attention-based U-Net on residual frame information with uncertainty."""

from __future__ import annotations

# Standard Library
from typing import override

# PyTorch
import torch
from lightning.pytorch.cli import LightningArgumentParser

# First party imports
from attention_unet import ResidualTwoPlusOneDataModule
from cli.common import I2RInternshipCommonCLI
from models.attention.uncertainty.lightning_module import (
    UncertaintyResidualAttentionLightningModule,
)
from models.two_plus_one import get_temporal_conv_type
from utils import utils
from utils.types import ResidualMode

BATCH_SIZE_TRAIN = 2  # Default batch size for training.
NUM_FRAMES = 5  # Default number of frames.
torch.set_float32_matmul_precision("medium")


class UncertaintyResidualAttentionCLI(I2RInternshipCommonCLI):
    """CLI class for residual attention with uncertainty task."""

    @override
    def before_instantiate_classes(self) -> None:
        """Run some code before instantiating the classes.

        Sets the torch multiprocessing mode depending on the optical flow method.
        """
        super().before_instantiate_classes()
        # GUARD: Check for subcommand
        if (subcommand := self.config.get("subcommand")) is not None:
            # GUARD: Check that residual_mode is set
            if (
                residual_mode := self.config.get(subcommand).get("residual_mode")
            ) is not None:
                # Set mp mode to `spawn` for OPTICAL_FLOW_GPU.
                if ResidualMode[residual_mode] == ResidualMode.OPTICAL_FLOW_GPU:
                    try:
                        torch.multiprocessing.set_start_method("spawn")
                        print("Multiprocessing mode set to `spawn`")
                        return
                    except RuntimeError as e:
                        raise RuntimeError(
                            "Cannot set multiprocessing mode to spawn"
                        ) from e
        print("Multiprocessing mode set as default.")

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        """Add extra arguments to CLI parser."""
        super().add_arguments_to_parser(parser)
        parser.add_argument("--residual_mode", help="Residual calculation mode")
        parser.link_arguments(
            "residual_mode", "model.residual_mode", compute_fn=utils.get_residual_mode
        )
        parser.link_arguments(
            "residual_mode", "data.residual_mode", compute_fn=utils.get_residual_mode
        )

        parser.add_argument(
            "temporal_conv_type",
            help="What kind of temporal convolutional layer to use.",
        )
        parser.link_arguments(
            "temporal_conv_type",
            "model.temporal_conv_type",
            compute_fn=get_temporal_conv_type,
        )

        default_arguments = self.default_arguments | {
            "image_loading_mode": "RGB",
            "dl_classification_mode": "MULTICLASS_MODE",
            "eval_classification_mode": "MULTICLASS_MODE",
            "residual_mode": "SUBTRACT_NEXT_FRAME",
            "model_architecture": "UNET",
            "trainer.max_epochs": 50,
            "model.encoder_name": "resnet50",
            "model.encoder_weights": "imagenet",
            "model.in_channels": 3,
            "model.classes": 4,
            "temporal_conv_type": "ORIGINAL",
        }

        parser.set_defaults(default_arguments)


if __name__ == "__main__":
    cli = UncertaintyResidualAttentionCLI(
        UncertaintyResidualAttentionLightningModule,
        ResidualTwoPlusOneDataModule,
        save_config_callback=None,
        auto_configure_optimizers=False,
        parser_kwargs={
            "fit": {
                "default_config_files": [
                    "./configs/uncertainty_residual_attention.yaml"
                ]
            },
            "predict": {
                "default_config_files": [
                    "./configs/uncertainty_residual_attention.yaml",
                    "./configs/predict.yaml",
                ]
            },
        },
    )
