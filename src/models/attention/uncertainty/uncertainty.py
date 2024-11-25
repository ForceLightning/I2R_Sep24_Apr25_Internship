# -*- coding: utf-8 -*-
"""Uncertainty modules for U-Net with attention mechanism."""

from __future__ import annotations

# Third-Party
from einops import rearrange

# PyTorch
import torch
from torch import Tensor, nn
from torch.nn.modules.dropout import _DropoutNd
from torch_uncertainty.models.wrappers.mc_dropout import _dropout_checks

# Local folders
from ..segmentation_model import ResidualAttentionUnet, ResidualAttentionUnetPlusPlus


class MCDropout(nn.Module):
    """MC Dropout wrapper for a model containing nn.Dropout modules."""

    def __init__(
        self,
        model: ResidualAttentionUnet | ResidualAttentionUnetPlusPlus,
        num_estimators: int,
        last_layer: bool = False,
        on_batch: bool = True,
    ) -> None:
        """Initialise the MC Dropout wrapper.

        Args:
            model: model to wrap
            num_estimators: number of estimators to use during the evaluation
            last_layer: whether to apply dropout to the last layer only.
            on_batch: Perform the MC-Dropout on the batch-size. Otherwise in a for loop.
                Useful when constrained in memory.

        Warning:
            This module will work only if you apply dropout through modules declared in
            the constructor (__init__).

        Warning:
            The `last-layer` option disables the lastly initialized dropout during
            evaluation: make sure that the last dropout is either functional or a
            module of its own.

        """
        super().__init__()
        filtered_modules = list(
            filter(lambda m: isinstance(m, _DropoutNd), model.modules())
        )
        if last_layer:
            filtered_modules = filtered_modules[-1:]

        _dropout_checks(filtered_modules, num_estimators)
        self.last_layer = last_layer
        self.on_batch = on_batch
        self.core_model = model
        self.num_estimators = num_estimators
        self.filtered_modules = filtered_modules

    def train(self, mode: bool = True) -> nn.Module:
        """Override the default train method to set the training mode.

        Set training mode of each submodule to be the same as the module itself except
        for the selected dropout modules.

        Args:
            mode: whether to set the module to training mode. Defaults to True.

        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        for module in self.filtered_modules:
            module.train()
        return self

    def forward(
        self,
        x_img: Tensor,
        x_res: Tensor,
    ) -> Tensor:
        """Forward pass of the model.

        During training, the forward pass is the same as of the core model.
        During evaluation, the forward pass is repeated `num_estimators` times
        either on the batch size or in a for loop depending on
        :attr:`last_layer`.

        Args:
            x_img: input tensor of shape (B, ...)
            x_res: input tensor of shape (B, ...)

        Returns:
            Tensor: output tensor of shape (:attr:`num_estimators` * B, ...)

        """
        bs = x_img.size(0)
        if self.training:
            return self.core_model(x_img, x_res)
        if self.on_batch:
            x_img = x_img.repeat(self.num_estimators, *([1] * (x_img.ndim - 1)))
            x_res = x_res.repeat(self.num_estimators, *([1] * (x_res.ndim - 1)))
            logits = self.core_model(x_img, x_res)
            logits = rearrange(logits, "(m b) c h w -> b m c h w", b=bs)
        # Else, for loop
        else:
            logits = torch.cat(
                [self.core_model(x_img, x_res) for _ in range(self.num_estimators)],
                dim=0,
            )
            logits = rearrange(logits, "(m b) c h w -> b m c h w", b=bs)

        return logits
