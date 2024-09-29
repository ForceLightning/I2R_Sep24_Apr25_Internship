# -*- coding: utf-8 -*-
"""Logging utilities for metrics."""
from typing import Literal

import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import functional as F
from torchmetrics import Metric, MetricCollection

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
        masks_preds = F.one_hot(masks_preds, num_classes=4).permute(0, -1, 1, 2)
        module.metrics[prefix].update(masks_preds, masks_one_hot)
    else:
        raise NotImplementedError(
            f"The mode {module.eval_classification_mode.name} is not implemented."
        )

    return masks_preds, masks_one_hot.bool()


def setup_metrics(module: L.LightningModule, metric: Metric | None, classes: int):
    """Sets up the metrics (dice scores) for the model.

    Args:
        module: The LightningModule instance.
        metric: The metric to use. If None, the default is the GeneralizedDiceScoreVariant.
        classes: The number of classes in the dataset.
    """

    # Create the metrics for the model.
    for stage in ["train", "val", "test"]:
        metric_weighted = (
            metric
            if metric
            else GeneralizedDiceScoreVariant(
                num_classes=classes,
                per_class=True,
                include_background=False,
                weight_type="linear",
                weighted_average=True,
            )
        )
        module.__setattr__(f"{stage}_metric_weighted", metric_weighted)

        metric_macro = GeneralizedDiceScoreVariant(
            num_classes=classes,
            per_class=True,
            include_background=False,
            weight_type="linear",
            weighted_average=True,
            return_type="macro_avg",
        )

        module.__setattr__(f"{stage}_metric_macro", metric_macro)

        metric_classes = GeneralizedDiceScoreVariant(
            num_classes=classes,
            per_class=True,
            include_background=False,
            weight_type="linear",
            weighted_average=True,
            return_type="per_class",
        )

        module.__setattr__(f"{stage}_metric_classes", metric_classes)

        metric_class_2_3_weighted = GeneralizedDiceScoreVariant(
            num_classes=classes,
            per_class=True,
            include_background=False,
            weight_type="linear",
            weighted_average=True,
            only_for_classes=[0, 0, 1, 1],
            return_type="weighted_avg",
        )

        module.__setattr__(
            f"{stage}_metric_class_2_3_weighted", metric_class_2_3_weighted
        )

        metric_class_2_3_macro = GeneralizedDiceScoreVariant(
            num_classes=classes,
            per_class=True,
            include_background=False,
            weight_type="linear",
            weighted_average=True,
            only_for_classes=[0, 0, 1, 1],
            return_type="macro_avg",
        )

        module.__setattr__(f"{stage}_metric_class_2_3_macro", metric_class_2_3_macro)

        # Combine the metrics into a single object.
        metric_combined = MetricCollection(
            {
                "dice_weighted_avg": metric_weighted,
                "dice_macro_avg": metric_macro,
                "dice_per_class": metric_classes,
                "dice_weighted_class_2_3": metric_class_2_3_weighted,
                "dice_macro_class_2_3": metric_class_2_3_macro,
            },
            prefix=f"{stage}/",
            compute_groups=True,
        )

        # NOTE: This allows for the metrics to be accessed via the module, and Lightning
        # will handle casting the metrics to the right dtype.
        module.__setattr__(f"{stage}_metric_combined", metric_combined)

        module.metrics[stage] = metric_combined


def shared_metric_logging_epoch_end(module: L.LightningModule, prefix: str):
    """Logs the metrics for the model. This is called at the end of the epoch.

    This method only handles the logging of Dice scores.

    Args:
        module: The LightningModule instance.
        prefix: The runtime mode (train, val, test).
    """
    metric_obj = module.metrics[prefix]

    if isinstance(metric_obj, GeneralizedDiceScoreVariant):
        # Handle the single metric case.
        _single_generalized_dice_logging(module, metric_obj, prefix)
    elif isinstance(metric_obj, MetricCollection):
        # Handle the grouped metric case.
        _grouped_generalized_dice_logging(module, metric_obj, prefix)


def _single_generalized_dice_logging(
    module: L.LightningModule, metric_obj: GeneralizedDiceScoreVariant, prefix: str
):
    """Logs the metrics for the model for a single GeneralizedDiceScoreVariant.

    Args:
        module: The LightningModule instance.
        metric_obj: The metric object.
        prefix: The runtime mode (train, val, test).
    """
    weighted_avg = metric_obj.compute()
    macro_avg = metric_obj.macro_avg_metric
    per_class = metric_obj.per_class_metric

    if isinstance(module.logger, TensorBoardLogger):
        module.log(f"hp/{prefix}/dice_weighted_avg", weighted_avg.item(), logger=True)
        module.log(f"hp/{prefix}/dice_macro_avg", macro_avg.item(), logger=True)

    module.log(f"{prefix}/dice_macro_avg", macro_avg.item(), logger=True)
    module.log(f"{prefix}/dice_weighted_avg", weighted_avg.item(), logger=True)

    for i, class_metric in enumerate(per_class):
        if i == 0:  # NOTE: Skips background class.
            continue
        module.log(f"{prefix}/dice_class_{i}", class_metric.item(), logger=True)
    metric_obj.reset()


def _grouped_generalized_dice_logging(
    module: L.LightningModule, metric_obj: MetricCollection, prefix: str
):
    """Logs the metrics for the model for a MetricCollection of
    GeneralizedDiceScoreVariant.

    Args:
        module: The LightningModule instance.
        metric_obj: The metric object.
        prefix: The runtime mode (train, val, test).
    """
    if all(
        isinstance(metric, GeneralizedDiceScoreVariant)
        for _, metric in metric_obj.items()
    ):
        results: dict[str, torch.Tensor] = metric_obj.compute()
        results_new: dict[str, torch.Tensor] = {}
        for k, v in results.items():
            if k in [
                "val/dice_macro_avg",
                "val/dice_macro_class_2_3",
                "val/dice_weighted_avg",
                "val/dice_weighted_class_2_3",
            ]:
                results_new[f"hp/{k}"] = v

        results.update(results_new)

        # HACK: This is a bit of a hack to turn the per_class metric into individual
        # metrics for each class. This is done to make it easier to log the metrics.
        if (per_class := results.get(f"{prefix}/dice_per_class", None)) is not None:
            assert isinstance(
                per_class, torch.Tensor
            ), f"Metric `per_class` is of an invalid type: {type(per_class)}"

            for i, class_metric in enumerate(per_class):
                if i == 0:
                    continue  # NOTE: Skips background class.
                results[f"{prefix}/dice_class_{i}"] = class_metric

            # Remove the per_class metric from the results.
            del results[f"{prefix}/dice_per_class"]

        # GUARD: Ensure that the metrics are of the correct type. Just use the basic
        # Python primatives and torch Tensors.
        assert all(
            isinstance(metric, float)
            or isinstance(metric, int)
            or isinstance(metric, bool)
            or isinstance(metric, torch.Tensor)
            for _, metric in results.items()
        ), f"Invalid metric primative type for dict: {results}"

        module.log_dict(results, on_step=False, on_epoch=True)
