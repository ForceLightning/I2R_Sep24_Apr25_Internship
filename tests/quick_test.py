# -*- coding: utf-8 -*-
"""Quick testing script for the project."""
import os
import ssl
from typing import Literal
from unittest import mock

import cv2
import matplotlib.pyplot as plt
import pytest
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose

from attention_unet import (
    ResidualAttentionCLI,
    ResidualAttentionUnetLightning,
    ResidualTwoPlusOneDataModule,
)
from cine import CineBaselineDataModule, CineCLI
from cine import LightningUnetWrapper as UnmodifiedUnet
from dataset.dataset import CineDataset, LGEDataset, TwoPlusOneDataset
from lge import LGECLI, LGEBaselineDataModule
from two_plus_one import TwoPlusOneCLI
from two_plus_one import TwoPlusOneDataModule as TwoPlusOneDataModule
from two_plus_one import TwoPlusOneUnetLightning as TwoPlusOneUnet
from two_stream import TwoStreamCLI, TwoStreamDataModule, TwoStreamUnetLightning
from utils.utils import ClassificationMode, LoadingMode, get_transforms

ssl._create_default_https_context = ssl._create_unverified_context


class TestTwoPlusOneCLI:
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

    def test_flat_conv(self):
        """
        Tests whether the ResNet50 (2+1) can train with a flat temporal convolutional
        layer.
        """
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


class TestLGECLI:
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


class TestTwoStreamCLI:
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
        """
        Tests whether the ResNet50 (TwoStream) can train on a single batch.
        """
        args = (
            self.default_train_args + self.default_resnet_args + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_senet_training(self):
        """
        Tests whether the SENet154 (TwoStream) can validate on a single batch.
        """
        args = (
            self.default_train_args + self.default_senet_args + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_quick_resnet_testing(self):
        """
        Tests whether the ResNet50 (TwoStream) can train on a single batch.
        """
        args = self.default_test_args + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_senet_testing(self):
        """
        Tests whether the SENet154 (TwoStream) can validate on a single batch.
        """
        args = self.default_test_args + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_resnet_rgb(self):
        """
        Tests whether the ResNet50 (TwoStream) can train on a single batch with RGB
        images.
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
        Tests whether the ResNet50 (TwoStream) can train on a single batch with greyscale
        images.
        """
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.greyscale_colour_mode
            + self.fast_dev_run_args
        )
        self._run_with_args(args)


class TestAttentionUnetCLI:
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
        """
        Tests whether the ResNet50 (AttentionUnet) can train on a single batch.
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
        Tests whether the SENet154 (AttentionUnet) can validate on a single batch.
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
        Tests whether the ResNet50 (AttentionUnet) can train on a single batch.
        """
        args = self.default_test_args + self.default_frames + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_senet_testing(self):
        """
        Tests whether the SENet154 (AttentionUnet) can validate on a single batch.
        """
        args = self.default_test_args + self.default_frames + self.fast_dev_run_args
        self._run_with_args(args)

    def test_quick_resnet_rgb(self):
        """
        Tests whether the ResNet50 (AttentionUnet) can train on a single batch with RGB
        images.
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
        Tests whether the ResNet50 (AttentionUnet) can train on a single batch with
        greyscale images.
        """
        args = (
            self.default_train_args
            + self.default_resnet_args
            + self.default_frames
            + self.greyscale_colour_mode
            + self.fast_dev_run_args
        )
        self._run_with_args(args)

    def test_flat_conv(self):
        """
        Tests whether the ResNet50 (AttentionUnet) can train with a flat temporal
        convolutional layer.
        """
        args = (
            self.default_train_args
            + self.default_resnet_args
            + ["--model.num_frames=30"]
            + self.greyscale_colour_mode
            + self.flatten_conv_args
            + self.fast_dev_run_args
        )
        self._run_with_args(args)


@pytest.mark.skip("Image loading process has changed a bit.")
class TestImageLoading:
    data_dir: str = "data/train_val/"
    test_dir: str = "data/test/"
    indices_dir: str = "data/indices/"
    frames: int = 10
    select_frame_method: Literal["specific", "consecutive"] = "specific"
    classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE
    _, transforms_mask, _ = get_transforms(LoadingMode.RGB)
    transforms_img = Compose(
        [
            v2.ToImage(),
            v2.Resize(224, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    transforms_together = Compose([v2.Identity()])
    lge_dataset = LGEDataset(
        img_dir=os.path.join(os.getcwd(), data_dir, "LGE"),
        mask_dir=os.path.join(os.getcwd(), data_dir, "masks"),
        idxs_dir=os.path.join(os.getcwd(), indices_dir),
        transform_img=transforms_img,
        transform_mask=transforms_mask,
        transform_together=transforms_together,
        classification_mode=classification_mode,
        combine_train_val=True,
    )
    cine_dataset = CineDataset(
        img_dir=os.path.join(os.getcwd(), data_dir, "Cine"),
        mask_dir=os.path.join(os.getcwd(), data_dir, "masks"),
        idxs_dir=os.path.join(os.getcwd(), indices_dir),
        transform_img=transforms_img,
        transform_mask=transforms_mask,
        transform_together=transforms_together,
        classification_mode=classification_mode,
        combine_train_val=True,
    )
    tpo_dataset: TwoPlusOneDataset = TwoPlusOneDataset(
        img_dir=os.path.join(os.getcwd(), data_dir, "Cine"),
        mask_dir=os.path.join(os.getcwd(), data_dir, "masks"),
        idxs_dir=os.path.join(os.getcwd(), indices_dir),
        frames=frames,
        select_frame_method=select_frame_method,
        transform_img=transforms_img,
        transform_mask=transforms_mask,
        transform_together=transforms_together,
        classification_mode=classification_mode,
        combine_train_val=True,
    )

    def _test_batched_image_loading(
        self, dataset: LGEDataset | CineDataset | TwoPlusOneDataset
    ):
        """
        Checks if batched image loading differs from directly loading with cv2.

        This basically checks if the dimensions are the same and if there is any
        rotation applied from numpy's transpose and torch's permute. If there is a
        difference, a plot of both images are shown.
        """
        im_tensor, mask, name = dataset[0]
        if len(mask.shape) != 2:
            raise ValueError(f"Mask of shape: {mask.shape} is invalid")
        im_a = im_tensor[0]
        im_tuple = cv2.imreadmulti(
            os.path.join(dataset.img_dir, name), flags=cv2.IMREAD_COLOR
        )
        img_list = im_tuple[1]
        im_b = self.transforms_together(self.transforms_img(img_list[0]))

        try:
            assert torch.allclose(
                im_a, im_b
            ), f"max difference of {(im_a - im_b).max()} detected"
        except AssertionError as e:
            print(im_a.shape, im_b.shape)
            _, ax = plt.subplots(1, 2)
            ax[0].imshow(im_a.permute(1, 2, 0))
            ax[0].set_title("From Dataset")
            ax[1].imshow(img_list[0])
            ax[1].set_title("From File")
            plt.show(block=True)
            raise e

    def test_batched_image_lge(self):
        """
        Checks if the batched image loading differs from directly loading with cv2.

        This test checks for the LGE dataset.
        """
        self._test_batched_image_loading(self.lge_dataset)

    def test_batched_image_cine(self):
        """
        Checks if the batched image loading differs from directly loading with cv2.

        This test checks for the Cine dataset.
        """
        self._test_batched_image_loading(self.cine_dataset)

    def test_batched_image_two_plus_one(self):
        """
        Checks if the batched image loading differs from directly loading with cv2.

        This test checks for the TwoPlusOne dataset.
        """
        self._test_batched_image_loading(self.tpo_dataset)
