# -*- coding: utf-8 -*-
"""Quick testing script for the project."""
import ssl
from unittest import mock

from attention_unet import (
    ResidualAttentionCLI,
    ResidualAttentionUnetLightning,
    ResidualTwoPlusOneDataModule,
)
from cine import CineBaselineDataModule, CineCLI
from cine import LightningUnetWrapper as UnmodifiedUnet
from lge import LGECLI, LGEBaselineDataModule
from two_plus_one import TwoPlusOneCLI
from two_plus_one import TwoPlusOneDataModule as TwoPlusOneDataModule
from two_plus_one import TwoPlusOneUnetLightning as TwoPlusOneUnet
from two_stream import TwoStreamCLI, TwoStreamDataModule, TwoStreamUnetLightning

ssl._create_default_https_context = ssl._create_unverified_context


class TestTwoPlusOneCLI:
    """Test integration of the TwoPlusOne CLI with the models and datasets."""

    default_train_args = [
        "fit",
        "--trainer.precision=bf16-mixed",
        "--model.classes=4",
        "--config",
        "./configs/two_plus_one.yaml",
        "--config",
        "./configs/cine_tpo_resnet50.yaml",
        "--data.num_workers=0",
        "--config",
        "./configs/no_checkpointing.yaml",
    ]
    default_frames = ["--model.num_frames=5"]
    default_test_args = [
        "test",
        "--config",
        "./configs/two_plus_one.yaml",
        "--config",
        "./configs/cine_tpo_resnet50.yaml",
        "--config",
        "./configs/testing.yaml",
        "--config",
        "./configs/no_checkpointing.yaml",
    ]
    default_senet_args = ["--model.encoder_name=senet154", "--data.batch_size=2"]
    default_resnet_args = ["--model.encoder_name=resnet50"]
    default_colour_mode = ["--image_loading_mode=RGB"]
    greyscale_colour_mode = ["--image_loading_mode=GREYSCALE", "--model.in_channels=1"]
    flatten_conv_args = ["--model.flat_conv=True"]
    fast_dev_run_args = ["--trainer.fast_dev_run=1"]
    filename = "two_plus_one.py"

    def _run_with_args(self, args: list[str]):
        with mock.patch("sys.argv", [self.filename] + args):
            TwoPlusOneCLI(
                TwoPlusOneUnet,
                TwoPlusOneDataModule,
                save_config_callback=None,
                auto_configure_optimizers=False,
            )

    def test_quick_resnet_training(self):
        """Tests whether the ResNet50 (2+1) can train on a single batch."""
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.default_frames
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_senet_training(self):
        """Tests whether the SENet154 (2+1) can validate on a single batch."""
        args = (
            self.default_train_args
            + self.default_senet_args
            + self.default_frames
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_testing(self):
        """Tests whether the ResNet50 (2+1) can train on a single batch."""
        args = self.default_test_args + self.default_frames + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_senet_testing(self):
        """Tests whether the SENet154 (2+1) can validate on a single batch."""
        args = self.default_test_args + self.default_frames + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_resnet_rgb(self):
        """Tests whether the ResNet50 (2+1) can train on a single batch with RGB images."""
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.default_frames
            + self.default_colour_mode
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_greyscale(self):
        """Tests whether the ResNet50 (2+1) can train on a single batch with greyscale images."""
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.default_frames
            + self.greyscale_colour_mode
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_flat_conv(self):
        """Tests whether the ResNet50 (2+1) can train with a flat temporal convolutional layer."""
        args = (
            self.default_train_args
            + self.default_resnet_args
            + ["--model.num_frames=30"]
            + self.greyscale_colour_mode
            + self.flatten_conv_args
            + self.fast_dev_run_args
        )
        self._run_with_args(args)


class TestCineCLI:
    """Test integration of the Cine CLI with the models and datasets."""

    default_train_args = [
        "fit",
        "--trainer.precision=bf16-mixed",
        "--model.classes=4",
        "--config",
        "./configs/cine.yaml",
        "--config",
        "./configs/cine_tpo_resnet50.yaml",
        "--trainer.logger=False",
        "--data.num_workers=0",
        "--config",
        "./configs/no_checkpointing.yaml",
    ]
    default_test_args = [
        "test",
        "--config",
        "./configs/cine.yaml",
        "--config",
        "./configs/cine_tpo_resnet50.yaml",
        "--config",
        "./configs/testing.yaml",
        "--trainer.logger=False",
        "--data.num_workers=0",
        "--config",
        "./configs/no_checkpointing.yaml",
    ]
    default_senet_args = ["--model.encoder_name=senet154", "--data.batch_size=2"]
    default_resnet_args = ["--model.encoder_name=resnet50"]
    fast_dev_run_args = ["--trainer.fast_dev_run=1"]
    default_colour_mode = ["--image_loading_mode=RGB"]
    greyscale_colour_mode = ["--image_loading_mode=GREYSCALE", "--model.in_channels=30"]
    filename = "cine.py"

    def _run_with_args(self, args: list[str]):
        with mock.patch("sys.argv", [self.filename] + args):
            CineCLI(
                UnmodifiedUnet,
                CineBaselineDataModule,
                save_config_callback=None,
                auto_configure_optimizers=False,
            )

    def test_quick_resnet_training(self):
        """Tests whether the ResNet50 (Cine) can train on a single batch."""
        args = (
            self.default_train_args + self.default_resnet_args + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_senet_training(self):
        """Tests whether the SENet154 (Cine) can validate on a single batch."""
        args = (
            self.default_train_args + self.default_senet_args + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_testing(self):
        """Tests whether the ResNet50 (Cine) can train on a single batch."""
        args = self.default_test_args + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_senet_testing(self):
        """Tests whether the SENet154 (Cine) can validate on a single batch."""
        args = self.default_test_args + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_resnet_rgb(self):
        """Tests whether the ResNet50 (Cine) can train on a single batch with RGB images."""
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.default_colour_mode
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_greyscale(self):
        """Tests whether the ResNet50 (Cine) can train on a single batch with greyscale images."""
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.greyscale_colour_mode
            + self.fast_dev_run_args
        )
        self._run_with_args(args)


class TestLGECLI:
    """Test integration of the LGE CLI with the models and datasets."""

    default_train_args = [
        "fit",
        "--trainer.precision=bf16-mixed",
        "--model.classes=4",
        "--config",
        "./configs/lge.yaml",
        "--trainer.logger=False",
        "--data.num_workers=0",
        "--config",
        "./configs/no_checkpointing.yaml",
    ]
    default_test_args = [
        "test",
        "--config",
        "./configs/lge.yaml",
        "--config",
        "./configs/testing.yaml",
        "--trainer.logger=False",
        "--data.num_workers=0",
        "--config",
        "./configs/no_checkpointing.yaml",
    ]
    default_senet_args = ["--model.encoder_name=senet154", "--data.batch_size=4"]
    default_resnet_args = ["--model.encoder_name=resnet50"]
    default_colour_mode = ["--image_loading_mode=RGB"]
    greyscale_colour_mode = ["--image_loading_mode=GREYSCALE", "--model.in_channels=1"]
    fast_dev_run_args = ["--trainer.fast_dev_run=1"]
    filename = "lge.py"

    def _run_with_args(self, args: list[str]):
        with mock.patch("sys.argv", [self.filename] + args):
            LGECLI(
                UnmodifiedUnet,
                LGEBaselineDataModule,
                save_config_callback=None,
                auto_configure_optimizers=False,
            )

    def test_quick_resnet_training(self):
        """Tests whether the ResNet50 (LGE) can train on a single batch."""
        args = (
            self.default_train_args + self.default_resnet_args + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_senet_training(self):
        """Tests whether the SENet154 (LGE) can validate on a single batch."""
        args = (
            self.default_train_args + self.default_senet_args + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_testing(self):
        """Tests whether the ResNet50 (LGE) can train on a single batch."""
        args = self.default_test_args + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_senet_testing(self):
        """Tests whether the SENet154 (LGE) can validate on a single batch."""
        args = self.default_test_args + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_resnet_rgb(self):
        """Tests whether the ResNet50 (LGE) can train on a single batch with RGB images."""
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.default_colour_mode
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_greyscale(self):
        """Tests whether the ResNet50 (LGE) can train on a single batch with greyscale images."""
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.greyscale_colour_mode
            + self.fast_dev_run_args
        )
        self._run_with_args(args)


class TestTwoStreamCLI:
    """Test integration of the TwoStream CLI with the models and datasets."""

    default_train_args = [
        "fit",
        "--trainer.precision=bf16-mixed",
        "--model.classes=4",
        "--config",
        "./configs/two_stream.yaml",
        "--trainer.logger=False",
        "--data.num_workers=0",
        "--config",
        "./configs/no_checkpointing.yaml",
    ]
    default_test_args = [
        "test",
        "--config",
        "./configs/two_stream.yaml",
        "--config",
        "./configs/testing.yaml",
        "--trainer.logger=False",
        "--data.num_workers=0",
        "--config",
        "./configs/no_checkpointing.yaml",
    ]
    default_senet_args = ["--model.encoder_name=senet154", "--data.batch_size=2"]
    default_resnet_args = ["--model.encoder_name=resnet50"]
    default_colour_mode = ["--image_loading_mode=RGB"]
    greyscale_colour_mode = ["--image_loading_mode=GREYSCALE", "--model.in_channels=1"]
    fast_dev_run_args = ["--trainer.fast_dev_run=1"]
    filename = "two_stream.py"

    def _run_with_args(self, args: list[str]):
        with mock.patch("sys.argv", [self.filename] + args):
            TwoStreamCLI(
                TwoStreamUnetLightning,
                TwoStreamDataModule,
                save_config_callback=None,
                auto_configure_optimizers=False,
            )

    def test_quick_resnet_training(self):
        """Tests whether the ResNet50 (TwoStream) can train on a single batch."""
        args = (
            self.default_train_args + self.default_resnet_args + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_senet_training(self):
        """Tests whether the SENet154 (TwoStream) can validate on a single batch."""
        args = (
            self.default_train_args + self.default_senet_args + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_testing(self):
        """Tests whether the ResNet50 (TwoStream) can train on a single batch."""
        args = self.default_test_args + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_senet_testing(self):
        """Tests whether the SENet154 (TwoStream) can validate on a single batch."""
        args = self.default_test_args + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_resnet_rgb(self):
        """Tests whether the ResNet50 (TwoStream) can train on a single batch with RGB images."""
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.default_colour_mode
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_greyscale(self):
        """Tests whether the ResNet50 (TwoStream) can train on a single batch with greyscale images."""
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.greyscale_colour_mode
            + self.fast_dev_run_args
        )
        self._run_with_args(args)


class TestAttentionUnetCLI:
    """Test integration of the AttentionUnet CLI with the models and datasets."""

    default_train_args = [
        "fit",
        "--trainer.precision=bf16-mixed",
        "--model.classes=4",
        "--config",
        "./configs/residual_attention.yaml",
        "--config",
        "./configs/cine_tpo_resnet50.yaml",
        "--data.num_workers=0",
        "--config",
        "./configs/no_checkpointing.yaml",
        "--data.batch_size=1",
    ]
    default_frames = ["--model.num_frames=5"]
    default_test_args = [
        "test",
        "--config",
        "./configs/residual_attention.yaml",
        "--config",
        "./configs/cine_tpo_resnet50.yaml",
        "--config",
        "./configs/testing.yaml",
        "--config",
        "./configs/no_checkpointing.yaml",
        "--data.batch_size=1",
    ]
    default_senet_args = ["--model.encoder_name=senet154", "--data.batch_size=1"]
    default_resnet_args = ["--model.encoder_name=resnet50"]
    default_colour_mode = ["--image_loading_mode=RGB"]
    greyscale_colour_mode = ["--config", "./configs/residual_attention_greyscale.yaml"]
    flatten_conv_args = ["--model.flat_conv=True"]
    fast_dev_run_args = ["--trainer.fast_dev_run=1"]
    filename = "attention_unet.py"

    def _run_with_args(self, args: list[str]):
        with mock.patch("sys.argv", [self.filename] + args):
            ResidualAttentionCLI(
                ResidualAttentionUnetLightning,
                ResidualTwoPlusOneDataModule,
                save_config_callback=None,
                auto_configure_optimizers=False,
            )

    def test_quick_resnet_training(self):
        """Tests whether the ResNet50 (AttentionUnet) can train on a single batch."""
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.default_frames
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_senet_training(self):
        """Tests whether the SENet154 (AttentionUnet) can validate on a single batch."""
        args = (
            self.default_train_args
            + self.default_senet_args
            + self.default_frames
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_testing(self):
        """Tests whether the ResNet50 (AttentionUnet) can train on a single batch."""
        args = self.default_test_args + self.default_frames + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_senet_testing(self):
        """Tests whether the SENet154 (AttentionUnet) can validate on a single batch."""
        args = self.default_test_args + self.default_frames + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_resnet_rgb(self):
        """Tests whether the ResNet50 (AttentionUnet) can train on a single batch with RGB images."""
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.default_frames
            + self.default_colour_mode
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_greyscale(self):
        """Tests whether the ResNet50 (AttentionUnet) can train on a single batch with greyscale images."""
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.default_frames
            + self.greyscale_colour_mode
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_flat_conv(self):
        """Tests whether the ResNet50 (AttentionUnet) can train with a flat temporal convolutional layer."""
        args = (
            self.default_train_args
            + self.default_resnet_args
            + ["--model.num_frames=30"]
            + self.greyscale_colour_mode
            + self.flatten_conv_args
            + self.fast_dev_run_args
        )
        self._run_with_args(args)
