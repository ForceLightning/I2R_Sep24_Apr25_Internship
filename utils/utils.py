from enum import Enum, auto
from typing import Sequence

import lightning as L
import torch
from torch.nn import functional as F
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


class ClassificationType(Enum):
    MULTICLASS_MODE = auto()
    MULTILABEL_MODE = auto()


def get_checkpoint_filename(version: str | None) -> str | None:
    if version is not None:
        return version + "-epoch={epoch}-step={step}"
    else:
        return version


def get_classification_mode(mode: str) -> ClassificationType:
    """Gets the classification mode from a string input.

    Raises:
        KeyError: If the mode is not an implemented mode.
    """
    return ClassificationType[mode]


def get_version_name(ckpt_path: str | None) -> str | None:
    if ckpt_path is not None:
        version = ckpt_path.split("-")[0]
        return version
    return None


def configure_optimizers(module: L.LightningModule):
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


def shared_metric_calculation(
    module: L.LightningModule,
    images: torch.Tensor,
    masks: torch.Tensor,
    masks_proba: torch.Tensor,
    prefix: str,
):
    bs = images.shape[0]
    metric: torch.Tensor
    if module.classification_mode == ClassificationType.MULTILABEL_MODE:
        masks_preds = masks_proba > 0.5  # BS x C x H x W
        metric = module.metric(masks_preds, masks)
    elif module.classification_mode == ClassificationType.MULTICLASS_MODE:
        # Output: BS x C x H x W
        masks_one_hot = F.one_hot(masks.squeeze(), num_classes=4).permute(0, -1, 1, 2)

        # Output: BS x C x H x W
        masks_preds = masks_proba.argmax(dim=1)
        masks_preds_one_hot = F.one_hot(masks_preds, num_classes=4).permute(0, -1, 1, 2)

        class_distribution = masks_one_hot.sum(dim=[0, 2, 3])  # 1 x C
        class_distribution = class_distribution.div(
            class_distribution[1:].sum()
        ).squeeze()

        metric = module.metric(masks_preds_one_hot, masks_one_hot)
        weighted_avg = metric[1:] @ class_distribution[1:]
        module.log(
            f"{prefix}_dice_(weighted_avg)",
            weighted_avg.item(),
            batch_size=bs,
            on_epoch=True,
        )
    else:
        raise NotImplementedError(
            f"The mode {module.classification_mode.name} is not implemented."
        )

    module.log(f"{prefix}_dice_(macro_avg)", metric.mean().item(), batch_size=bs)

    for i, class_metric in enumerate(metric.detach().cpu()):
        if i == 0:  # NOTE: Skips background class.
            continue
        module.log(
            f"{prefix}_dice_class_{i}",
            class_metric.item(),
            batch_size=bs,
        )

    return masks_preds
