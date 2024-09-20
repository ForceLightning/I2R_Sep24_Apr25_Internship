# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Callable, Literal, override

import torch
from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.base.initialization import (
    initialize_decoder,
    initialize_head,
)
from segmentation_models_pytorch.encoders import get_encoder

from models.common import UnetDecoder


class TwoStreamSegmentationModel(SegmentationModel):

    @override
    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            initialize_head(self.classification_head)

    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, lge: torch.Tensor, cine: torch.Tensor
    ) -> torch.Tensor:
        added_features = []

        # The first layer of the skip connection gets ignored, but in order for the
        # indexing later on to work, the feature output needs an empty first output.
        added_features.append(["EMPTY"])

        # Go through each frame of the image and add the output features to a list.
        lge_features = self.lge_encoder(lge)

        cine_features = self.cine_encoder(cine)

        # Goes through each layer and gets the LGE and Cine output from that layer then
        # adds them element-wise.
        for index in range(1, 6):
            lge_output = lge_features[index]
            cine_output = cine_features[index]

            added_output = torch.add(cine_output, lge_output)
            added_features.append(added_output)

        # Send the added features up the decoder.
        decoder_output = self.decoder(*added_features)
        masks = self.segmentation_head(decoder_output)

        return masks


class TwoStreamUnet(TwoStreamSegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: list[int] = [256, 128, 64, 32, 16],
        decoder_attention_type: Literal["scse"] | None = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: str | Callable | None = None,
        num_frames: int = 30,
        aux_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.lge_encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.cine_encoder = get_encoder(
            encoder_name,
            in_channels=in_channels * num_frames,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.lge_encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            centre=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.lge_encoder.out_channels[-1],
                **aux_params,
            )

        else:
            self.classification_head = None

        self.name = f"u-{encoder_name}"
        self.initialize()
