# -*- coding: utf-8 -*-
"""Utility functions for the project."""
from typing import override

import lightning as L
import torch
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, OneCycleLR
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose
from warmup_scheduler import GradualWarmupScheduler

from utils.types import ClassificationMode, LoadingMode, ResidualMode


class LightningGradualWarmupScheduler(LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer."""

    def __init__(
        self,
        optimizer: Optimizer,
        multiplier: int = 2,
        total_epoch: int = 5,
        T_max=50,
        after_scheduler=None,
    ):
        """Init the scheduler.

        Args:
            optimizer: Wrapped optimizer.
            multiplier: The multiplier for the learning rate.
            total_epoch: The total number of epochs for the warm-up phase.
            T_max: The maximum number of epochs for the cosine annealing scheduler.
            after_scheduler: The scheduler to run after the warm-up phase.

        """
        self.optimizer = optimizer
        after_scheduler = (
            after_scheduler if after_scheduler else CosineAnnealingLR(optimizer, T_max)
        )
        self.scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=multiplier,
            total_epoch=total_epoch,
            after_scheduler=after_scheduler,
        )

    @override
    def step(self, epoch=None, metrics=None):
        return self.scheduler.step(epoch, metrics)


def get_classification_mode(mode: str) -> ClassificationMode:
    """Get the classification mode from a string input.

    Args:
        mode: The classification mode

    Raises:
        KeyError: If the mode is not an implemented mode.

    """
    return ClassificationMode[mode]


def get_residual_mode(mode: str) -> ResidualMode:
    """Get the residual calculation mode from a string input.

    Args:
        mode: The residual calculation mode.

    Raises:
        KeyError: If the mode is not an implemented mode.

    """
    return ResidualMode[mode]


def get_loading_mode(mode: str) -> LoadingMode:
    """Get the classification mode from a string input.

    Args:
        mode: The classification mode

    Raises:
        KeyError: If the mode is not an implemented mode.

    """
    return LoadingMode[mode]


def get_checkpoint_filename(version: str | None) -> str | None:
    """Get the checkpoint filename with the version name if it exists.

    Args:
        version: The version name of the model.

    """
    if version is None:
        return None
    return version + "-epoch={epoch}-step={step}-val_loss={val/loss:.4f}"


def get_best_weighted_avg_dice_filename(version: str | None) -> str:
    """Get the filename for the best weighted average dice score.

    Args:
        version: The version name of the model.

    """
    suffix = "-epoch={epoch}-step={step}-dice_w_avg={val/dice_weighted_avg:.4f}"
    if version is None:
        return suffix
    return version + suffix


def get_best_macro_avg_dice_class_2_3_filename(version: str | None) -> str:
    """Get the filename for the best macro average dice score for class 2 & 3.

    Args:
        version: The version name of the model.

    """
    suffix = "-epoch={epoch}-step={step}-dice_m_avg_2_3={val/dice_macro_class_2_3:.4f}"
    if version is None:
        return suffix
    return version + suffix


def get_last_checkpoint_filename(version: str | None) -> str | None:
    """Get the filename for the last checkpoint.

    Args:
        version: The version name of the model.

    """
    if version is not None:
        return version + "-last"
    else:
        return version


def get_version_name(ckpt_path: str | None) -> str | None:
    """Get the version name from the checkpoint path.

    Args:
        ckpt_path: The path to the checkpoint.

    """
    if ckpt_path is not None:
        version = ckpt_path.split("-")[0]
        return version
    return None


def configure_optimizers(module: L.LightningModule):
    """Configure the optimizer and learning rate scheduler for the model.

    Args:
        module: The LightningModule instance.

    """
    module.optimizer_kwargs.update({"lr": module.learning_rate})
    match module.optimizer:
        case "adam":
            module.optimizer_kwargs.update({"fused": True})
            optimizer = Adam(
                params=module.model.parameters(), **module.optimizer_kwargs
            )
        case "adamw":
            module.optimizer_kwargs.update({"fused": True})
            optimizer = AdamW(
                params=module.model.parameters(), **module.optimizer_kwargs
            )
        case "sgd":
            optimizer = SGD(params=module.model.parameters(), **module.optimizer_kwargs)
        case _:
            raise NotImplementedError(f"optimizer {module.optimizer} not implemented!")

    if isinstance(module.scheduler, str):
        match module.scheduler:
            case "gradual_warmup_scheduler":
                kwargs = {
                    "optimizer": optimizer,
                    "multiplier": module.multiplier,
                    "total_epoch": max(module.total_epochs // 10, 1),
                    "T_max": module.total_epochs,
                }
                kwargs |= module.scheduler_kwargs
                scheduler = {
                    "scheduler": LightningGradualWarmupScheduler(**kwargs),
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss",
                    "strict": True,
                }
            case "one_cycle":
                # Defaults
                kwargs = {
                    "optimizer": optimizer,
                    "max_lr": module.learning_rate,
                    "total_steps": module.trainer.estimated_stepping_batches,
                    "pct_start": 0.1,
                }
                # Apply hyperparameters
                kwargs |= module.scheduler_kwargs

                scheduler = {
                    "scheduler": OneCycleLR(**kwargs),
                    "interval": "step",
                }
            case "cosine_anneal":
                kwargs = {
                    "optimizer": optimizer,
                    "T_max": module.trainer.estimated_stepping_batches,
                }
                kwargs |= module.scheduler_kwargs

                scheduler = {
                    "scheduler": CosineAnnealingLR(**kwargs),
                    "interval": "step",
                }
            case _:
                raise NotImplementedError(
                    f"Scheduler of type {module.scheduler} not implemented"
                )
    else:
        scheduler = module.scheduler(optimizer, **module.scheduler_kwargs)
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def get_transforms(
    loading_mode: LoadingMode, augment: bool = False
) -> tuple[Compose, Compose, Compose]:
    """Get default transformations for all datasets.

    The default implementation resizes the images to (224, 224), casts them to float32,
    normalises them, and sets them to greyscale if the loading mode is not RGB.

    Args:
        loading_mode: The loading mode for the images.
        augment: Whether to augment the images and masks together.

    Returns:
        tuple: The image, mask, combined, and final resize transformations

    """
    # Sets the image transforms
    transforms_img = Compose(
        [
            v2.ToImage(),
            v2.Resize(224, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            v2.Identity() if loading_mode == LoadingMode.RGB else v2.Grayscale(1),
        ]
    )

    # Sets the mask transforms
    transforms_mask = Compose(
        [
            v2.ToImage(),
            v2.Resize(224, antialias=True),
            v2.ToDtype(torch.long, scale=False),
        ]
    )

    # Randomly rotates +/- 180 deg and warps the image.
    transforms_together = Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(
                180.0,  # pyright: ignore[reportArgumentType]
                v2.InterpolationMode.BILINEAR,
            ),
            v2.ElasticTransform(alpha=33.0),
        ]
        if augment
        else [v2.Identity()]
    )

    return transforms_img, transforms_mask, transforms_together


def get_accumulate_grad_batches(batch_size: int):
    """Get the number of batches to accumulate the gradients.

    Args:
        batch_size: The batch size for training.

    Returns:
        int: The number of batches to accumulate the gradients

    """
    if batch_size >= 8:
        return 1
    else:
        return 8 // batch_size
