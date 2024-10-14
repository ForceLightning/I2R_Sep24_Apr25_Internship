# -*- coding: utf-8 -*-
"""2+1D U-Net model."""
from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Literal, override

import torch
from segmentation_models_pytorch.base.heads import ClassificationHead, SegmentationHead
from segmentation_models_pytorch.base.model import SegmentationModel
from segmentation_models_pytorch.decoders.unet.model import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn

from models.common import ENCODER_OUTPUT_SHAPES


class OneD(nn.Module):
    """1D Temporal Convolutional Block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_frames: int,
        flat: bool = False,
        activation: str | type[nn.Module] | None = None,
    ) -> None:
        """Init the 1D Temporal Convolutional Block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_frames: Number of frames in the input tensor.
            flat: If True, only one convolutional layer is used.
            activation: Activation function to use.

        Raises:
            NotImplementedError: If the number of frames is not implemented.

        Note:
            The number of frames must be one of 5, 10, 15, 20, or 30.

        """
        super().__init__()
        self.activation: type[nn.Module]
        if isinstance(activation, type):
            self.activation = activation
        else:
            match activation:
                case "relu":
                    self.activation = nn.ReLU
                case "gelu":
                    self.activation = nn.GELU
                case "mish":
                    self.activation = nn.Mish
                case "elu":
                    self.activation = nn.ELU
                case "silu" | "swish":
                    self.activation = nn.SiLU
                case _:
                    warnings.warn(
                        f"Activation function {activation} not recognized. Using ReLU.",
                        stacklevel=2,
                    )
                    self.activation = nn.ReLU

        if flat:
            self.one = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=num_frames,
                    stride=num_frames,
                    padding=0,
                ),
                self.activation(),
            )
        else:
            kernels: list[int] = []
            match num_frames:
                case 5:
                    kernels = [5]
                case 10:
                    kernels = [5, 2]
                case 15:
                    kernels = [5, 3]
                case 20:
                    kernels = [5, 4]
                case 25:
                    kernels = [5, 5]
                case 30:
                    kernels = [5, 3, 2]
                case _:
                    raise NotImplementedError(
                        f"Model with num_frames of {num_frames} not implemented!"
                    )

            layers: list[nn.Module] = []
            for i, k in enumerate(kernels):
                layers += [
                    nn.Conv1d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=k,
                        stride=k,
                        padding=0,
                    ),
                    self.activation(),
                ]
            self.one = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.one(x)


class DilatedOneD(nn.Module):
    """1D Temporal Convolutional Block with dilations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_frames: int,
        sequence_length: int,
        flat: bool = False,
        activation: str | type[nn.Module] | None = None,
    ):
        """Init the 1D Temporal Convolutional Block with dilations.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_frames: Number of frames in the input tensor.
            sequence_length: Length of the sequence.
            flat: If True, only one convolutional layer is used.
            activation: Activation function to use.

        Raises:
            NotImplementedError: If the number of frames is not implemented.

        Note:
            The number of frames must be one of 5, 10, 15, 20, or 30.

        """
        super().__init__()
        if isinstance(activation, type):
            self.activation = activation
        else:
            match activation:
                case "relu":
                    self.activation = nn.ReLU
                case "gelu":
                    self.activation = nn.GELU
                case "mish":
                    self.activation = nn.Mish
                case "elu":
                    self.activation = nn.ELU
                case "silu" | "swish":
                    self.activation = nn.SiLU
                case _:
                    warnings.warn(
                        f"Activation function {activation} not recognized. Using ReLU.",
                        stacklevel=2,
                    )
                    self.activation = nn.ReLU
        if flat:
            self.one = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=num_frames,
                    stride=num_frames,
                    padding=0,
                ),
                self.activation(),
            )
        else:
            kernels: list[int] = []
            match num_frames:
                case 5:
                    kernels = [5]
                case 10:
                    kernels = [5, 2]
                case 15:
                    kernels = [5, 3]
                case 20:
                    kernels = [5, 4]
                case 25:
                    kernels = [5, 5]
                case 30:
                    kernels = [5, 3, 2]
                case _:
                    raise NotImplementedError(
                        f"Model with num_frames of {num_frames} not implemented!"
                    )

            layers: list[nn.Module] = []
            for i, k in enumerate(kernels):
                multiplication_factor = num_frames // math.prod(kernels[: i + 1])
                dilation = sequence_length * multiplication_factor
                layers += [
                    nn.Conv1d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=k,
                        stride=1,
                        padding=0,
                        dilation=dilation,
                    ),
                    self.activation(),
                ]

            self.one = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.one(x)


def compress_2(stacked_outputs: torch.Tensor, block: OneD) -> torch.Tensor:
    """Apply the OneD temporal convolution on the stacked outputs.

    Args:
        stacked_outputs: 5D tensor of shape (num_frames, batch_size, num_channels, h, w).
        block: 1d temporal convolutional block.

    Return:
        torch.Tensor: 4D tensor of shape (batch_size, num_channels, h, w).

    """
    # Input shape: (B, F, C, H, W).
    b, f, c, h, w = stacked_outputs.shape
    # Reshape to: (B, H, W, C, F).
    inputs = stacked_outputs.permute(0, 3, 4, 2, 1).contiguous()
    # Inputs to a Conv1D must be of shape (N, C_in, L_in).
    inputs = inputs.view(b * h * w * c, 1, f)
    # Outputs are of shape (B * H * W * C, C_out)
    out = block(inputs)
    # Take the mean over the channel dimension -> (B * H * W * C, 1) and squeeze
    out = out.mean(dim=1).squeeze(dim=1)
    # Return outputs to shape (B, H, W, C) -> (B, C, H, W)
    final_out = out.view(b, h, w, c).permute(0, 3, 1, 2)

    return final_out


def compress_dilated(stacked_outputs: torch.Tensor, block: DilatedOneD) -> torch.Tensor:
    """Apply the DilatedOneD temporal convolution on the stacked outputs.

    Args:
        stacked_outputs: 5D tensor of shape (num_frames, batch_size, num_channels, h, w).
        block: 1d temporal convolutional block.

    Return:
        torch.Tensor: 4D tensor of shape (batch_size, num_channels, h, w).

    """
    # Input shape: (B, F, C, H, W).
    b, f, c, h, w = stacked_outputs.shape
    # Reshape to: (B, C, F, H, W).
    inputs = stacked_outputs.permute(0, 2, 1, 3, 4).contiguous()
    # Inputs to a Conv1D must be of shape (N, C_in, L_in).
    # Reshape to (B * C, 1, F * H * W)
    inputs = inputs.view(b * c, 1, f * h * w)
    # Outputs are of shape (B * C, C_out, H * W)
    out = block(inputs)
    # Take the mean over the channel dimension -> (B * C, 1, H * W) and squeeze.
    out = out.mean(dim=1).squeeze(dim=1)
    # Return outputs of shape (B, C, H, W)
    final_out = out.view(b, c, h, w)

    return final_out


class TwoPlusOneUnet(SegmentationModel):
    """2+1D U-Net model."""

    _default_decoder_channels = [256, 128, 64, 32, 16]
    _default_skip_conn_channels = [2, 5, 10, 20, 40]

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: list[int] = _default_decoder_channels,
        decoder_attention_type: Literal["scse"] | None = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: str | type[nn.Module] | None = None,
        skip_conn_channels: list[int] = _default_skip_conn_channels,
        num_frames: Literal[5, 10, 15, 20, 30] = 5,
        aux_params: dict[str, Any] | None = None,
        flat_conv: bool = False,
        res_conv_activation: str | None = None,
        use_dilations: bool = False,
    ) -> None:
        """Init the 2+1D U-Net model.

        Args:
            encoder_name: Name of the encoder.
            encoder_depth: Depth of the encoder.
            encoder_weights: Weights to use for the encoder.
            decoder_use_batchnorm: If True, use batch normalization in the decoder.
            decoder_channels: Number of channels in the decoder.
            decoder_attention_type: Attention type to use in the decoder.
            in_channels: Number of input channels.
            classes: Number of classes.
            activation: Activation function to use. This can be a string or a class to
                be instantiated.
            skip_conn_channels: Number of channels in each skip connection's temporal
                convolutions.
            num_frames: Number of frames in the input tensor.
            aux_params: Auxiliary parameters for the model.
            flat_conv: If True, only one convolutional layer is used.
            res_conv_activation: Activation function to use in the U-Net.
            use_dilations: If True, use dilated convolutions in the temporal
                convolutions.

        """
        super().__init__()
        self.num_frames = num_frames
        self.flat_conv = flat_conv
        self.activation = activation
        self.use_dilations = use_dilations
        self.encoder_name = encoder_name
        self.res_conv_activation = res_conv_activation
        self.skip_conn_channels = skip_conn_channels

        # Define encoder, decoder, segmentation head and classification head.
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=encoder_name.startswith("vgg"),
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
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = f"u-{encoder_name}"
        self.compress: Callable[..., torch.Tensor]
        self.initialize()

    @override
    def initialize(self) -> None:
        """Initialize the model.

        This method initializes the decoder and the segmentation head. It also
        initializes the 1D temporal convolutional blocks with the correct number of
        output channels for each layer of the encoder.
        """
        # Define encoders and segmentation head.
        super().initialize()

        # Create the 1D temporal conv blocks, with more output channels the more pixels
        # the output has.
        # At the first 1D layer, there will only be 49 pixels (7 x 7), so fewer output
        # channels are needed.
        # Ideally, every step should require x4 channels. Due to memory limitations,
        # this was not possible.

        # INFO: Parameter tuning for output channels.
        onedlayers: list[OneD | DilatedOneD] = []
        for i, out_channels in enumerate(self.skip_conn_channels):
            mod: OneD | DilatedOneD
            if self.use_dilations and self.num_frames in [5, 30]:
                _resnet_out_channels, h, w = ENCODER_OUTPUT_SHAPES[self.encoder_name][i]
                mod = DilatedOneD(
                    1,
                    out_channels,
                    self.num_frames,
                    h * w,
                    flat=self.flat_conv,
                    activation=self.res_conv_activation,
                )
                self.compress = compress_dilated
            else:
                mod = OneD(
                    1,
                    out_channels,
                    self.num_frames,
                    self.flat_conv,
                    self.res_conv_activation,
                )
                self.compress = compress_2
            onedlayers.append(mod)

        self.onedlayers = nn.ModuleList(onedlayers)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: 5D tensor of shape (batch_size, num_frames, channels, height, width).

        Return:
            torch.Tensor: 4D tensor of shape (batch_size, classes, height, width).

        """
        # The first layer of the skip connection gets ignored, but in order for the
        # indexing later on to work, the feature output needs an empty first output.
        compressed_features: list[torch.Tensor | list[str]] = [["EMPTY"]]

        # Go through each frame of the image and add the output features to a list.
        features_list: list[torch.Tensor] = []

        assert x.numel() != 0, f"Input tensor is empty: {x}"

        # NOTE: This goes through by batch actually.
        for img in x:
            self.check_input_shape(img)
            features = self.encoder(img)
            features_list.append(features)

        # Goes through each layer and gets the output from that layer from all the
        # feature outputs.
        # PERF: Maybe this can be done in parallel?
        for index in range(1, 6):
            layer_output = []
            for outputs in features_list:
                image_output = outputs[index]
                layer_output.append(image_output)
            layer_output = torch.stack(layer_output)

            # Define the 1D block with the correct number of output channels.
            block: OneD | DilatedOneD = self.onedlayers[
                index - 1
            ]  # pyright: ignore[reportAssignmentType] False positive

            # Applies the compress_2 function to resize and reorder channels.
            compressed_output = self.compress(layer_output, block)
            compressed_features.append(compressed_output)

        # Send the compressed features up the decoder
        decoder_output = self.decoder(*compressed_features)

        # Apply segmentation head and return the prediction
        masks = self.segmentation_head(decoder_output)
        return masks

    @override
    @torch.no_grad()
    def predict(self, x):
        """Inference method.

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            torch.Tensor: 4D torch tensor with shape (batch_size, classes, height,
            width).

        """
        if self.training:
            self.eval()

        x = self.foward(x)
        return x
