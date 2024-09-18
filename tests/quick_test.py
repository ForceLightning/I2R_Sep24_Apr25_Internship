# -*- coding: utf-8 -*-
"""Quick testing script for the project."""
import os
import unittest
from unittest import mock

import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2
from torchvision.transforms.transforms import Compose

from cine import CineBaselineDataModule, CineCLI
from cine import LightningUnetWrapper as UnmodifiedUnet
from dataset.dataset import TwoPlusOneDataset
from lge import LGECLI, LGEBaselineDataModule
from two_plus_one import TwoPlusOneCLI
from two_plus_one import TwoPlusOneDataModule as TwoPlusOneDataModule
from two_plus_one import UnetLightning as TwoPlusOneUnet
from utils.utils import ClassificationMode


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
    default_colour_mode = ["--image_loading_mode=RGB"]
    greyscale_colour_mode = ["--image_loading_mode=GREYSCALE", "--model.in_channels=1"]
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
        """
        Tests whether the ResNet50 (2+1) can train on a single batch.
        """
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.default_frames
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_senet_training(self):
        """
        Tests whether the SENet154 (2+1) can validate on a single batch.
        """
        args = (
            self.default_train_args
            + self.default_senet_args
            + self.default_frames
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_testing(self):
        """
        Tests whether the ResNet50 (2+1) can train on a single batch.
        """
        args = self.default_test_args + self.default_frames + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_senet_testing(self):
        """
        Tests whether the SENet154 (2+1) can validate on a single batch.
        """
        args = self.default_test_args + self.default_frames + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_resnet_rgb(self):
        """
        Tests whether the ResNet50 (2+1) can train on a single batch with RGB images.
        """
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.default_frames
            + self.default_colour_mode
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_greyscale(self):
        """
        Tests whether the ResNet50 (2+1) can train on a single batch with greyscale
        images.
        """
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.default_frames
            + self.greyscale_colour_mode
            + self.fast_dev_run_args
        )
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
        """
        Tests whether the ResNet50 (Cine) can train on a single batch.
        """
        args = (
            self.default_train_args + self.default_resnet_args + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_senet_training(self):
        """
        Tests whether the SENet154 (Cine) can validate on a single batch.
        """
        args = (
            self.default_train_args + self.default_senet_args + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_testing(self):
        """
        Tests whether the ResNet50 (Cine) can train on a single batch.
        """
        args = self.default_test_args + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_senet_testing(self):
        """
        Tests whether the SENet154 (Cine) can validate on a single batch.
        """
        args = self.default_test_args + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_resnet_rgb(self):
        """
        Tests whether the ResNet50 (Cine) can train on a single batch with RGB images.
        """
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.default_colour_mode
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_greyscale(self):
        """
        Tests whether the ResNet50 (Cine) can train on a single batch with greyscale
        images.
        """
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.greyscale_colour_mode
            + self.fast_dev_run_args
        )
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
        """
        Tests whether the ResNet50 (LGE) can train on a single batch.
        """
        args = (
            self.default_train_args + self.default_resnet_args + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_senet_training(self):
        """
        Tests whether the SENet154 (LGE) can validate on a single batch.
        """
        args = (
            self.default_train_args + self.default_senet_args + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_testing(self):
        """
        Tests whether the ResNet50 (LGE) can train on a single batch.
        """
        args = self.default_test_args + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_senet_testing(self):
        """
        Tests whether the SENet154 (LGE) can validate on a single batch.
        """
        args = self.default_test_args + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_resnet_rgb(self):
        """
        Tests whether the ResNet50 (LGE) can train on a single batch with RGB images.
        """
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.default_colour_mode
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_greyscale(self):
        """
        Tests whether the ResNet50 (LGE) can train on a single batch with greyscale
        images.
        """
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.greyscale_colour_mode
            + self.fast_dev_run_args
        )
        self._run_with_args(args)


class TestImageLoading(unittest.TestCase):
    data_dir: str = "data/train_val-20240905T025601Z-001/train_val/"
    test_dir: str = "data/test-20240905T012341Z-001/test/"
    indices_dir: str = "data/indices/"
    frames: int = 10
    select_frame_method = "specific"
    classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE
    transforms_img = Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    transforms_mask = Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    dataset: TwoPlusOneDataset = TwoPlusOneDataset(
        os.path.join(os.getcwd(), data_dir, "Cine"),
        os.path.join(os.getcwd(), data_dir, "masks"),
        os.path.join(os.getcwd(), indices_dir),
        frames=frames,
        select_frame_method=select_frame_method,
        transform_1=transforms_img,
        transform_2=transforms_mask,
        classification_mode=classification_mode,
    )

    def test_batched_image_loading(self):
        """
        Checks if batched image loading differs from directly loading with cv2.

        This basically checks if the dimensions are the same and if there is any
        rotation applied from numpy's transpose and torch's permute. If there is a
        difference, a plot of both images are shown.
        """
        im_tensor, _, name = self.dataset[0]
        im_a = im_tensor[0]
        im_tuple = cv2.imreadmulti(
            os.path.join(self.dataset.img_dir, name), flags=cv2.IMREAD_COLOR
        )
        img_list = im_tuple[1]
        im_b = self.transforms_img(img_list[0])

        try:
            assert torch.allclose(im_a, im_b)
        except AssertionError as e:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(im_a.permute(1, 2, 0))
            ax[0].set_title("From Dataset")
            ax[1].imshow(img_list[0])
            ax[1].set_title("From File")
            plt.show(block=True)
            raise e


if __name__ == "__main__":
    unittest.main()
