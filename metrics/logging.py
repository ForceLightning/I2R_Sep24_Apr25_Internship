from typing import Literal

import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import functional as F

from metrics.dice import GeneralizedDiceScoreVariant
from utils.utils import ClassificationMode


@torch.no_grad()
def shared_metric_calculation(
    module: L.LightningModule,
    masks: torch.Tensor,
    masks_proba: torch.Tensor,
    prefix: Literal["train", "val", "test"] = "train",
):
    """Calculates the metrics for the model.

    Args:
        module: The LightningModule instance.
        images: The input images.
        masks: The ground truth masks.
        masks_proba: The predicted masks.
        prefix: The runtime mode (train, val, test).
    """
    masks_one_hot = F.one_hot(masks.squeeze(dim=1), num_classes=4).permute(0, -1, 1, 2)

    # HACK: I'd be lying if I said otherwise. This checks the 4 possibilities (for now)
    # of the combinations of classification modes and sets the metrics correctly.
    if module.eval_classification_mode == ClassificationMode.MULTILABEL_MODE:
        masks_preds = masks_proba > 0.5  # BS x C x H x W
        if module.dl_classification_mode == ClassificationMode.MULTICLASS_MODE:
            module.metrics[prefix].update(masks_preds, masks_one_hot)
        else:
            module.metrics[prefix].update(masks_preds, masks)
    elif module.eval_classification_mode == ClassificationMode.MULTICLASS_MODE:
        # Output: BS x C x H x W
        masks_preds = masks_proba.argmax(dim=1)
        masks_preds_one_hot = F.one_hot(masks_preds, num_classes=4).permute(0, -1, 1, 2)
        module.metrics[prefix].update(masks_preds_one_hot, masks_one_hot)
    else:
        raise NotImplementedError(
            f"The mode {module.eval_classification_mode.name} is not implemented."
        )

    return masks_preds, masks_one_hot.bool()


def shared_metric_logging_epoch_end(module: L.LightningModule, prefix: str):
    if isinstance(module.metrics[prefix], GeneralizedDiceScoreVariant):
        metric = module.metrics[prefix].compute()
        macro_avg = module.metrics[prefix].macro_avg_metric
        per_class = module.metrics[prefix].per_class_metric

        if isinstance(module.logger, TensorBoardLogger):
            module.log(f"hp/{prefix}/dice_weighted_avg", metric.item(), logger=True)
            module.log(f"hp/{prefix}/dice_macro_avg", macro_avg.item(), logger=True)

        module.log(f"{prefix}/dice_macro_avg", macro_avg.item(), logger=True)
        module.log(f"{prefix}/dice_weighted_avg", metric.item(), logger=True)

        for i, class_metric in enumerate(per_class):
            if i == 0:  # NOTE: Skips background class.
                continue
            module.log(f"{prefix}/dice_class_{i}", class_metric.item(), logger=True)
        module.metrics[prefix].reset()
