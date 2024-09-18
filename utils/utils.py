# -*- coding: utf-8 -*-
"""Utility functions for the project."""
from enum import Enum, auto
from typing import Sequence

import lightning as L
import torch
from torch.optim._multi_tensor import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.optim.optimizer import Optimizer
from torchvision.transforms import Normalize
from warmup_scheduler import GradualWarmupScheduler


class InverseNormalize(Normalize):
    """Inverses the normalization and returns the reconstructed images in the input."""

    def __init__(
        self,
        mean: Sequence[float | int],
        std: Sequence[float | int],
    ):
        mean_tensor = torch.as_tensor(mean)
        std_tensor = torch.as_tensor(std)
        std_inv = 1 / (std_tensor + 1e-7)
        mean_inv = -mean_tensor * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return super().__call__(tensor.clone())


class LightningGradualWarmupScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        multiplier: int = 2,
        total_epoch: int = 5,
        T_max=50,
        after_scheduler=None,
    ):
        """Gradually warm-up(increasing) learning rate in optimizer then run
        after_scheduler.

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

    def step(self, epoch=None, metrics=None):
        return self.scheduler.step(epoch, metrics)


class ClassificationMode(Enum):
    """The classification mode for the model.

    MULTICLASS_MODE: The model is trained to predict a single class for each pixel.
    MULTILABEL_MODE: The model is trained to predict multiple classes for each pixel.
    """

    MULTICLASS_MODE = auto()
    MULTILABEL_MODE = auto()


def get_classification_mode(mode: str) -> ClassificationMode:
    """Gets the classification mode from a string input.

    Args:
        mode: The classification mode

    Raises:
        KeyError: If the mode is not an implemented mode.
    """
    return ClassificationMode[mode]


class LoadingMode(Enum):
    """Determines the image loading mode for the dataset.

    RGB: The images are loaded in RGB mode.
    GREYSCALE: The images are loaded in greyscale mode.
    """

    RGB = auto()
    GREYSCALE = auto()


def get_loading_mode(mode: str) -> LoadingMode:
    """Gets the classification mode from a string input.

    Args:
        mode: The classification mode

    Raises:
        KeyError: If the mode is not an implemented mode.
    """
    return LoadingMode[mode]


def get_checkpoint_filename(version: str | None) -> str | None:
    """Gets the checkpoint filename with the version name if it exists.

    Args:
        version: The version name of the model.
    """
    if version is None:
        return None
    return version + "-{epoch}-{step}"


def get_best_weighted_avg_dice_filename(version: str | None) -> str:
    """Gets the filename for the best weighted average dice score.

    Args:
        version: The version name of the model.
    """
    suffix = "-{epoch}-{step}-{val/dice_weighted_avg:.4f}"
    if version is None:
        return suffix
    return version + suffix


def get_last_checkpoint_filename(version: str | None) -> str | None:
    """Gets the filename for the last checkpoint.

    Args:
        version: The version name of the model.
    """
    if version is not None:
        return version + "-last"
    else:
        return version


def get_version_name(ckpt_path: str | None) -> str | None:
    """Gets the version name from the checkpoint path.

    Args:
        ckpt_path: The path to the checkpoint.
    """
    if ckpt_path is not None:
        version = ckpt_path.split("-")[0]
        return version
    return None


def configure_optimizers(module: L.LightningModule):
    """Configures the optimizer and learning rate scheduler for the model.

    Args:
        module: The LightningModule instance.
    """
    module.optimizer_kwargs.update({"lr": module.learning_rate})
    match module.optimizer:
        case "adam":
            optimizer = Adam(
                params=module.model.parameters(), **module.optimizer_kwargs
            )
        case "adamw":
            optimizer = AdamW(
                params=module.model.parameters(), **module.optimizer_kwargs
            )
        case _:
            raise NotImplementedError(f"optimizer {module.optimizer} not implemented!")

    if isinstance(module.scheduler, str):
        match module.scheduler:
            case "gradual_warmup_scheduler":
                module.scheduler_kwargs.update(
                    {
                        "optimizer": optimizer,
                        "multiplier": module.multiplier,
                        "total_epoch": 5,
                        "T_max": module.total_epochs,
                    }
                )
                scheduler = {
                    "scheduler": LightningGradualWarmupScheduler(
                        **module.scheduler_kwargs
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss",
                    "strict": True,
                }
            case _:
                raise NotImplementedError(
                    f"Scheduler of type {module.scheduler} not implemented"
                )
    else:
        scheduler = module.scheduler(optimizer, **module.scheduler_kwargs)
    return {"optimizer": optimizer, "lr_scheduler": scheduler}
