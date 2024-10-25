from __future__ import annotations

import math
from functools import partial
from typing import Any, Callable, Literal, Mapping, override

import numpy as np
import torch
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.encoders._base import EncoderMixin
from torch import nn
from torch.nn import functional as F


def conv_S(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    padding: int | tuple[int, int, int] = 1,
):
    """Initialise spatial convolutional layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Convolutional filter stride.
        padding: Padding shape.

    Return:
        nn.Conv3d: Initalised spatial convolutional layer.

    """
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=(1, 3, 3),
        stride=stride,
        padding=padding,
        bias=False,
    )


def conv_T(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    padding: int | tuple[int, int, int] = 1,
):
    """Initialise temporal convolutional layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Convolutional filter stride.
        padding: Padding shape.

    Return:
        nn.Conv3d: Initalised temporal convolutional layer.

    """
    return nn.Conv3d(
        in_channels, out_channels, (3, 1, 1), stride=stride, padding=padding, bias=False
    )


def downsample_basic_block(x: torch.Tensor, channels: int, stride: int) -> torch.Tensor:
    """Downsamples input with average pooling.

    Args:
        x: Input tensor.
        channels: Number of output channels.
        stride: Stride of average pooling layer.

    Return:
        torch.Tensor: Downsampled input.

    """
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_padding = torch.zeros(
        (out.size(0), channels - out.size(1), out.size(2), out.size(3), out.size(4)),
        requires_grad=True,
    )
    zero_padding = zero_padding.to(out.device)

    out = torch.cat([out.data, zero_padding], dim=1)
    return out


class Bottleneck(nn.Module):
    """Bottleneck layer."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | Callable[[torch.Tensor], torch.Tensor] | None = None,
        n_s: int = 0,
        depth_3d: int = 47,
        ST_struc: tuple[str, str, str] = ("A", "B", "C"),
    ):
        """Initialise the bottleneck.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride of the convolutional layer.
            downsample: Downsample layer.
            n_s: Current layer index.
            depth_3d: Number of 3D layers.
            ST_struc: Spatial-temporal structure.

        """
        super().__init__()
        self.downsample = downsample
        self.depth_3d = depth_3d
        self.ST_struc = ST_struc
        self.len_ST = len(self.ST_struc)

        stride_p = stride

        if self.downsample is not None:
            stride_p = (1, 2, 2)

        if n_s < self.depth_3d:
            if n_s == 0:
                stride_p = 1
            self.conv1 = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, bias=False, stride=stride_p
            )
            self.bn1 = nn.BatchNorm3d(out_channels)

        else:
            if n_s == self.depth_3d:
                stride_p = 2
            else:
                stride_p = 1

            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False, stride=stride_p
            )
            self.bn1 = nn.BatchNorm2d(out_channels)

        self.id = n_s
        self.ST = list(self.ST_struc)[self.id % self.len_ST]

        if self.id < self.depth_3d:
            self.conv2 = conv_S(out_channels, out_channels, stride=1, padding=(0, 1, 1))
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.conv3 = conv_T(out_channels, out_channels, stride=1, padding=(1, 0, 0))
            self.bn3 = nn.BatchNorm3d(out_channels)

        else:
            self.conv_normal = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.bn_normal = nn.BatchNorm2d(out_channels)

        if n_s < self.depth_3d:
            self.conv4 = nn.Conv3d(
                out_channels, out_channels * 4, kernel_size=1, bias=False
            )
            self.bn4 = nn.BatchNorm3d(out_channels * 4)
        else:
            self.conv4 = nn.Conv2d(
                out_channels, out_channels * 4, kernel_size=1, bias=False
            )
            self.bn4 = nn.BatchNorm2d(out_channels * 4)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def ST_A(self, x: torch.Tensor) -> torch.Tensor:
        """A-type spatial-temporal structure.

        Args:
            x: Input tensor.

        Return:
            torch.Tensor: Output tensor

        """
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x

    def ST_B(self, x: torch.Tensor) -> torch.Tensor:
        """B-type spatial-temporal structure.

        Args:
            x: Input tensor.

        Return:
            torch.Tensor: Output tensor

        """
        spat_x = self.conv2(x)
        spat_x = self.bn2(spat_x)
        spat_x = self.relu(spat_x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x + spat_x

    def ST_C(self, x: torch.Tensor) -> torch.Tensor:
        """C-type spatial-temporal structure.

        Args:
            x: Input tensor.

        Return:
            torch.Tensor: Output tensor

        """
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        tmp_x = self.conv3(x)
        tmp_x = self.bn3(x)
        tmp_x = self.relu(x)

        return x + tmp_x

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.id < self.depth_3d:  # C3D Parts
            match self.ST:
                case "A":
                    out = self.ST_A(out)
                case "B":
                    out = self.ST_B(out)
                case "C":
                    out = self.ST_C(out)
                case _:
                    raise NotImplementedError(f"P3D{self.ST} not implemented!")

        else:
            out = self.conv_normal(out)
            out = self.bn_normal(out)
            out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample is not None:
            res = self.downsample(x)

        out += res
        out = self.relu(out)

        return out


class TemporalSqueeze(nn.Module):
    """Squeeze temporal dimension."""

    @override
    def __init__(self):
        super().__init__()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sizes = x.size()
        x = x.view(-1, sizes[1], sizes[3], sizes[4])
        return x


class P3D(nn.Module):
    """P3D model."""

    def __init__(
        self,
        block: type[Bottleneck],
        layers: list[int],
        modality: Literal["RGB", "Greyscale", "Flow"] = "RGB",
        shortcut_type: Literal["A", "B", "C"] = "B",
        num_classes: int = 400,
        dropout: float = 0.5,
        ST_struc: tuple[
            Literal["A", "B", "C"], Literal["A", "B", "C"], Literal["A", "B", "C"]
        ] = ("A", "B", "C"),
    ):
        """Initialise the P3D model.

        Args:
            block: Block type.
            layers: Number of layers.
            modality: Input modality.
            shortcut_type: Shortcut type.
            num_classes: Number of classes.
            dropout: Dropout rate.
            ST_struc: Spatial-temporal structure.

        """
        super().__init__()
        self.inplanes = 64
        match modality:
            case "RGB":
                self.in_channels = 3
            case "Greyscale":
                self.in_channels = 1
            case "Flow":
                self.in_channels = 2

        self.ST_struc = ST_struc
        self.conv1_custom = nn.Conv3d(
            self.in_channels,
            64,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        )
        self.depth_3d = sum(
            layers[:3]
        )  # C3D layers are only (res2, res3, res4), res5 is C2D

        self.bn1 = nn.BatchNorm3d(64)
        self.cnt = 0
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0, 1, 1))
        self.maxpool_2 = nn.MaxPool3d(
            kernel_size=(2, 1, 1), padding=0, stride=(2, 1, 1)
        )

        blocks = []
        for i, layer in enumerate(layers[:4]):
            blocks.append(
                self._make_layer(
                    block, 2 ** (6 + i), layer, shortcut_type, stride=2 if i > 0 else 1
                )
            )

        self.squeeze = TemporalSqueeze()

        self.layers = nn.ModuleList(blocks)

        self.avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kenel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Some private attributes
        self._input_size = (self.in_channels, 16, 160, 160)
        self._input_mean = [0.485, 0.456, 0.406] if modality == "RGB" else [0.5]
        self._input_std = (
            [0.229, 0.224, 0.225]
            if modality == "RGB"
            else [np.mean([0.229, 0.224, 0.225])]
        )

    @property
    def scale_size(self):
        """Get the scale size of the model."""
        return (
            self.input_size[2] * 256 // 160
        )  # Assume that raw images are resized (360, 256)

    @property
    def temporal_length(self):
        """Get the temporal length of the model."""
        return self.input_size[1]

    @property
    def crop_size(self):
        """Get the crop size of the model."""
        return self.input_size[2]

    def _make_layer(
        self,
        block: type[Bottleneck],
        planes: int,
        blocks: int,
        shortcut_type: Literal["A", "B", "C"],
        stride: int = 1,
    ) -> nn.Sequential:
        """Make a layer.

        Args:
            block: Block type.
            planes: Number of planes.
            blocks: Number of blocks.
            shortcut_type: Shortcut type.
            stride: Stride.

        Return:
            nn.Sequential: Layer.

        """
        downsample = None
        stride_p = stride  # Especially for downsample branch.

        if self.cnt < self.depth_3d:
            if self.cnt == 0:
                stride_p = 1
            else:
                stride_p = (1, 2, 2)

            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == "A":
                    downsample = partial(
                        downsample_basic_block,
                        channels=planes * block.expansion,
                        stride=stride,
                    )
                else:
                    downsample = nn.Sequential(
                        nn.Conv3d(
                            self.inplanes,
                            planes * block.expansion,
                            kernel_size=1,
                            stride=stride_p,
                            bias=False,
                        ),
                        nn.BatchNorm3d(planes * block.expansion),
                    )

        else:
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == "A":
                    downsample = partial(
                        downsample_basic_block,
                        channels=planes * block.expansion,
                        stride=stride,
                    )
                else:
                    downsample = nn.Sequential(
                        nn.Conv2d(
                            self.inplanes,
                            planes * block.expansion,
                            kernel_size=1,
                            stride=2,
                            bias=False,
                        ),
                        nn.BatchNorm2d(planes * block.expansion),
                    )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                n_s=self.cnt,
                depth_3d=self.depth_3d,
                ST_struc=self.ST_struc,
            )
        )
        self.cnt += 1
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    n_s=self.cnt,
                    depth_3d=self.depth_3d,
                    ST_struc=self.ST_struc,
                )
            )
            self.cnt += 1

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.maxpool_2(self.layers[0](x))  # Part Res2
        x = self.maxpool_2(self.layers[1](x))  # Part Res3
        x = self.maxpool_2(self.layers[2](x))  # Part Res4

        x = self.squeeze(x)
        x = self.layers[3](x)
        x = self.avgpool(x)

        x = x.view(-1, self.fc.in_features)
        x = self.fc(self.dropout(x))

        return x

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


class P3DEncoder(P3D, EncoderMixin):
    """P3D encoder."""

    def __init__(self, out_channels: tuple[int, ...], depth: int = 5, **kwargs):
        """Initialise the P3D encoder.

        Args:
            out_channels: Number of output channels.
            depth: Depth of the encoder.
            **kwargs: Additional arguments.

        """
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = self.in_channels

        del self.fc
        del self.avgpool

    @override
    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1_custom, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layers[0], self.maxpool_2),
            nn.Sequential(self.layers[1], self.maxpool_2),
            nn.Sequential(self.layers[2], self.maxpool_2),
            nn.Sequential(self.squeeze, self.layers[3]),
        ]

    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, x: torch.Tensor
    ) -> list[torch.Tensor]:
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    @override
    def load_state_dict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, state_dict: Mapping[str, Any], **kwargs
    ):
        state_dict.pop("fc.bias", None)  # pyright: ignore[reportAttributeAccessIssue]
        state_dict.pop("fc.weight", None)  # pyright: ignore[reportAttributeAccessIssue]
        super().load_state_dict(state_dict, **kwargs)

    @override
    def set_in_channels(self, in_channels: int, pretrained: bool = True):
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        patch_first_conv(model=self, new_in_channels=in_channels, pretrained=pretrained)


def patch_first_conv(
    model: P3DEncoder,
    new_in_channels: int,
    default_in_channels: int = 3,
    pretrained: bool = True,
) -> None:
    """Patch first convolutional layer.

    Args:
        model: Model to patch.
        new_in_channels: New number of input channels.
        default_in_channels: Default number of input channels.
        pretrained: Whether the model is pretrained.

    """
    # Get first conv3d
    for module in model.modules():
        if isinstance(module, nn.Conv3d) and module.in_channels == default_in_channels:
            weight = module.weight.detach()
            module.in_channels = new_in_channels

            if not pretrained:
                module.weight = nn.parameter.Parameter(
                    torch.Tensor(
                        module.out_channels,
                        new_in_channels // module.groups,
                        *module.kernel_size,
                    )
                )
                module.reset_parameters()
            elif new_in_channels == 1:
                new_weight = weight.sum(1, keepdim=True)
                module.weight = nn.parameter.Parameter(new_weight)

            else:
                new_weight = torch.Tensor(
                    module.out_channels,
                    new_in_channels // module.groups,
                    *module.kernel_size,
                )
                for i in range(new_in_channels):
                    new_weight[:, i] = weight[:, i % default_in_channels]

                new_weight = new_weight * (default_in_channels / new_in_channels)
                module.weight = nn.parameter.Parameter(new_weight)
            break


def get_encoder(
    in_channels: int,
    output_stride: int = 32,
    encoder_name: Literal["p3d63", "p3d131", "p3d199"] = "p3d63",
    modality: Literal["RGB", "Greyscale", "Flow"] = "RGB",
    pretrained: bool = True,
    **kwargs,
) -> P3D:
    """Get the P3D encoder model.

    Args:
        in_channels: Number of input channels.
        output_stride: Output stride.
        encoder_name: Encoder name.
        modality: Input modality.
        pretrained: Whether the model is pretrained.
        **kwargs: Additional arguments.

    Return:
        P3DEncoder: P3D encoder model

    """
    match encoder_name:
        case "p3d63":
            model = P3D(Bottleneck, [3, 4, 6, 3], modality=modality, **kwargs)
        case "p3d131":
            model = P3D(Bottleneck, [3, 4, 23, 3], modality=modality, **kwargs)
        case "p3d199":
            model = P3D(Bottleneck, [3, 8, 36, 3], modality=modality, **kwargs)
            if pretrained:
                pretrained_file = None
                match modality:
                    case "Flow":
                        pretrained_file = "p3d_flow_199.checkpoint.pth.tar"
                    case _:
                        pretrained_file = "p3d_rgb_199.checkpoint.pth.tar"

                if pretrained_file:
                    weights = torch.load(pretrained_file)["state_dict"]
                    model.load_state_dict(weights)

    model.set_in_channels(in_channels, pretrained=pretrained)

    if output_stride != 32:
        model.make_dilated(output_stride)

    return model


class P3DUnet(SegmentationModel):
    """P3D UNet model."""

    def __init__(
        self,
        encoder_name: Literal["p3d63", "p3d131", "p3d199"] = "p3d63",
        modality: Literal["RGB", "Greyscale", "Flow"] = "RGB",
        decoder_use_batchnorm: bool = True,
        decoder_channels: list[int] | None = None,
        decoder_attention_type: Literal["scse"] | None = None,
        classes: int = 1,
        activation: str | type[nn.Module] | None = None,
        num_frames: Literal[5, 10, 15, 20, 30] = 5,
        aux_params: dict[str, Any] | None = None,
    ):
        """Initialise the P3D UNet model.

        Args:
            encoder_name: Encoder name.
            modality: Input modality.
            decoder_use_batchnorm: Whether to use batch normalisation in the decoder.
            decoder_channels: Number of channels in the decoder.
            decoder_attention_type: Attention type in the decoder.
            classes: Number of classes.
            activation: Activation function.
            num_frames: Number of frames.
            aux_params: Auxiliary parameters.

        """
        super().__init__()
        self.encoder_name = encoder_name
        self.modality = modality
        match modality:
            case "RGB":
                self.in_channels = 3
            case "Greyscale":
                self.in_channels = 1
            case "Flow":
                self.in_channels = 2
        self.num_frames = num_frames
        self.activation = activation

        # Handle defaults
        decoder_channels = (
            decoder_channels if decoder_channels else [256, 128, 64, 32, 16]
        )

        # TODO: Complete this.
        raise NotImplementedError("P3D UNet not implemented!")
