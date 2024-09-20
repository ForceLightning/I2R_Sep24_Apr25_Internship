from __future__ import annotations

from typing import Literal, Sequence, override

import torch
from segmentation_models_pytorch.base import modules as md
from torch import nn
from torch.nn import functional as F


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        attention_type: Literal["scse"] | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = md.Conv2dReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    @override
    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CentreBlock(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, use_batchnorm: bool = True
    ) -> None:
        conv1 = md.Conv2dReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: list[int],
        n_blocks: int = 5,
        use_batchnorm: bool = True,
        attention_type: Literal["scse"] | None = None,
        centre: bool = False,
    ) -> None:
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but decoder_channels is provided for {len(decoder_channels)} blocks."
            )

        # Remove first skip with same spatial resolution.
        encoder_channels = encoder_channels[1:]
        # Reverse channels to start from head of encoder.
        encoder_channels = encoder_channels[::-1]

        # Computing blocks input and output channels.
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if centre:
            self.centre = CentreBlock(
                in_channels=head_channels,
                out_channels=head_channels,
                use_batchnorm=use_batchnorm,
            )
        else:
            self.centre = nn.Identity()

        # Combine decoder keyword arguments.
        blocks = [
            DecoderBlock(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
                use_batchnorm=use_batchnorm,
                attention_type=attention_type,
            )
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    @override
    def forward(self, *features: Sequence[torch.Tensor]) -> torch.Tensor:
        features = features[1:]  # Remove the first skip with the same spatial res.
        features = features[::-1]  # Reverse channels to start with the head of the enc.

        head = features[0]
        skips = features[1:]

        x = self.centre(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
