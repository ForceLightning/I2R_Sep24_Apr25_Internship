# -*- coding: utf-8 -*-
"""Quick testing script for the project."""
import unittest
from unittest import mock

from cine import CineBaselineDataModule, CineCLI
from cine import LightningUnetWrapper as UnmodifiedUnet
from lge import LGECLI, LGEBaselineDataModule
from two_plus_one import CineDataModule as TwoPlusOneDataModule
from two_plus_one import TwoPlusOneCLI
from two_plus_one import UnetLightning as TwoPlusOneUnet


class TestTwoPlusOneCLI(unittest.TestCase):
    default_train_args = [
        "fit",
        "--trainer.precision=bf16-mixed",
        "--version=unittest",
        "--data.num_workers=0",
    ]
    default_frames = ["--model.num_frames=5"]
    default_test_args = ["test", "--version=unittest", "--data.num_workers=0"]
    default_senet_args = ["--model.encoder_name=senet154"]
    default_resnet_args = ["--model.encoder_name=resnet50"]
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
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.default_frames
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_senet_training(self):
        args = (
            self.default_train_args
            + self.default_senet_args
            + self.default_frames
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_testing(self):
        args = self.default_test_args + self.default_frames + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_senet_testing(self):
        args = self.default_test_args + self.default_frames + self.fast_dev_run_args
        self._run_with_args(args)


class TestCineCLI(unittest.TestCase):
    default_train_args = [
        "fit",
        "--trainer.precision=bf16-mixed",
        "--version=unittest",
        "--data.num_workers=0",
    ]
    default_test_args = ["test", "--version=unittest", "--data.num_workers=0"]
    default_senet_args = ["--model.encoder_name=senet154"]
    default_resnet_args = ["--model.encoder_name=resnet50"]
    fast_dev_run_args = ["--trainer.fast_dev_run=1"]
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
        args = (
            self.default_train_args + self.default_resnet_args + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_senet_training(self):
        args = (
            self.default_train_args + self.default_senet_args + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_testing(self):
        args = self.default_test_args + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_senet_testing(self):
        args = self.default_test_args + self.fast_dev_run_args
        self._run_with_args(args)


class TestLGECLI(unittest.TestCase):
    default_train_args = [
        "fit",
        "--trainer.precision=bf16-mixed",
        "--version=unittest",
        "--data.num_workers=0",
    ]
    default_test_args = ["test", "--version=unittest", "--data.num_workers=0"]
    default_senet_args = ["--model.encoder_name=senet154"]
    default_resnet_args = ["--model.encoder_name=resnet50"]
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
        args = (
            self.default_train_args + self.default_resnet_args + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_senet_training(self):
        args = (
            self.default_train_args + self.default_senet_args + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_testing(self):
        args = self.default_test_args + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_senet_testing(self):
        args = self.default_test_args + self.fast_dev_run_args
        self._run_with_args(args)


if __name__ == "__main__":
    unittest.main()
