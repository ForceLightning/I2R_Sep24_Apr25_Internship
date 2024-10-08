from __future__ import annotations

from typing import Literal, OrderedDict, override

import torch
from segmentation_models_pytorch.encoders._base import EncoderMixin
from torch import nn
from torch.nn.common_types import _size_3_t

from models.common import ENCODER_OUTPUT_SHAPES


class TSCSEModule(nn.Module):
    def __init__(
        self,
        in_depth: int,
        in_channels: int,
        height: int,
        width: int,
        reduction: int = 16,
    ):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

        self.sSE = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, height, width)),
            nn.Conv3d(in_channels, 1, (in_depth, 1, 1)),
            nn.Sigmoid(),
        )

        self.tSE = nn.Sequential(
            nn.AdaptiveAvgPool3d((in_depth, 1, 1)),
            nn.Conv3d(in_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cSE(x) + x * self.sSE(x) + x * self.tSE(x)


class Bottleneck(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.layers(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.tscse(out) + residual
        out = self.relu(out)

        return out


class TSCSEBottleneck(Bottleneck):
    expansion: int = 4

    def __init__(
        self,
        in_depth: int,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        groups: int,
        reduction: int,
        stride: _size_3_t = 1,
        downsample: nn.Module | None = None,
    ):
        conv1 = nn.Conv3d(
            in_channels, out_channels * 2, kernel_size=1, bias=False, stride=stride
        )
        bn1 = nn.BatchNorm3d(out_channels * 2)
        conv2 = nn.Conv3d(
            out_channels * 2,
            out_channels * 4,
            kernel_size=(1, 3, 3),
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        bn2 = nn.BatchNorm2d(out_channels * 4)
        conv3 = nn.Conv3d(out_channels * 4, out_channels * 4, kernel_size=1, bias=False)
        bn3 = nn.BatchNorm3d(out_channels * 4)

        self.layers = nn.Sequential(conv1, bn1, conv2, bn2, conv3, bn3)

        self.relu = nn.ReLU(inplace=True)
        self.tscse = TSCSEModule(in_depth, out_channels * 4, height, width, reduction)
        self.downsample = downsample
        self.stride = stride


class TSCSEResNetBottleneck(Bottleneck):
    expansion: int = 4

    def __init__(
        self,
        in_depth: int,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        groups: int,
        reduction: int,
        stride: _size_3_t = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, bias=False, stride=stride
        )
        bn1 = nn.BatchNorm3d(out_channels)
        conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            groups=groups,
            bias=False,
        )
        bn2 = nn.BatchNorm2d(out_channels)
        conv3 = nn.Conv3d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        bn3 = nn.BatchNorm3d(out_channels * 4)

        self.layers = nn.Sequential(conv1, bn1, conv2, bn2, conv3, bn3)

        self.relu = nn.ReLU(inplace=True)
        self.tscse = TSCSEModule(in_depth, out_channels * 4, height, width, reduction)
        self.downsample = downsample
        self.stride = stride


class TSCSENet(nn.Module):
    def __init__(
        self,
        name: Literal[
            "tscsenet154", "tscse_resnet50", "tscse_resnet101", "tscse_resnet152"
        ],
        block: type[TSCSEBottleneck] | type[TSCSEResNetBottleneck],
        depth: int,
        layers: list[int],
        groups: int,
        reduction: int,
        dropout_p: float = 0.2,
        inplanes: int = 128,
        input_3x3: bool = True,
        downsample_kernel_size: int = 3,
        downsample_padding: int = 1,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.inplanes = inplanes
        self.name: Literal[
            "tscsenet154", "tscse_resnet50", "tscse_resnet101", "tscse_resnet152"
        ] = name
        if input_3x3:
            layer0_modules = [
                (
                    "conv1",
                    nn.Conv3d(
                        3,
                        64,
                        (1, 3, 3),
                        stride=(1, 2, 2),
                        padding=(0, 1, 1),
                        bias=False,
                    ),
                ),
                ("bn1", nn.BatchNorm3d(64)),
                ("relu1", nn.ReLU(inplace=True)),
                (
                    "conv2",
                    nn.Conv3d(
                        64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1), bias=False
                    ),
                ),
                ("bn2", nn.BatchNorm3d(64)),
                ("relu2", nn.ReLU(inplace=True)),
                (
                    "conv3",
                    nn.Conv3d(
                        64, inplanes, (1, 3, 3), stride=1, padding=(0, 1, 1), bias=False
                    ),
                ),
                ("bn3", nn.BatchNorm3d(inplanes)),
                ("relu", nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                (
                    "conv1",
                    nn.Conv3d(
                        3,
                        inplanes,
                        kernel_size=(1, 7, 7),
                        stride=(1, 2, 2),
                        padding=(1, 3, 3),
                        bias=False,
                    ),
                ),
                ("bn1", nn.BatchNorm3d(inplanes)),
                ("relu1", nn.ReLU(inplace=True)),
            ]

        layer0_modules.append(
            ("pool", nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), ceil_mode=True))
        )
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0,
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=(1, 2, 2),
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=(1, 2, 2),
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[2],
            stride=(1, 2, 2),
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
        )
        self.avg_pool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)
        self.depth = depth

    def _make_layer(
        self,
        block: type[TSCSEBottleneck] | type[TSCSEResNetBottleneck],
        planes: int,
        blocks: int,
        groups: int,
        reduction: int,
        stride: _size_3_t = 1,
        downsample_kernel_size: _size_3_t = 1,
        downsample_padding: _size_3_t = 0,
    ):
        downsample = None
        if (
            (not isinstance(stride, tuple) and stride != 1)
            or (isinstance(stride, tuple) and stride[1] != 1 and stride[2] != 1)
            or self.inplanes != planes * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=downsample_kernel_size,
                    stride=stride,
                    padding=downsample_padding,
                    bias=False,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.depth,
                self.inplanes,
                planes,
                height=224,
                width=224,
                groups=groups,
                reduction=reduction,
                stride=stride,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            _, height, width = ENCODER_OUTPUT_SHAPES[self.name][i]
            layers.append(
                block(
                    self.depth,
                    self.inplanes,
                    planes,
                    height,
                    width,
                    groups,
                    reduction,
                    stride,
                    downsample,
                )
            )

        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.logits(x)

        return x


def replace_strides_with_dilation_3d(module: nn.Module, dilation_rate: int):
    for mod in module.modules():
        if isinstance(mod, nn.Conv3d):
            mod.stride = (1, 1, 1)
            mod.dilation = (1, dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = (0, (kh // 2) * dilation_rate, (kw // 2) * dilation_rate)


class Encoder3DMixin(EncoderMixin):
    _output_stride: _size_3_t = (1, 32, 32)

    @override
    def make_dilated(self, output_stride: _size_3_t):
        if (
            isinstance(output_stride, tuple)
            and output_stride[1] == 16
            and output_stride[2] == 16
            or (isinstance(output_stride, int) and output_stride == 16)
        ):
            stage_list = [5]
            dilation_list = [2]
        elif (
            isinstance(output_stride, tuple)
            and output_stride[1] == 8
            and output_stride[2] == 8
        ) or (isinstance(output_stride, int) and output_stride == 8):
            stage_list = [4, 5]
            dilation_list = [2, 4]
        else:
            raise ValueError(
                f"Output stride should be 16 / (1, 16, 16) or 8 / (1, 8, 8), got {output_stride}"
            )

        self._output_stride = output_stride

        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            replace_strides_with_dilation_3d(
                module=stages[stage_indx], dilation_rate=dilation_rate
            )


class TSCSENetEncoder(TSCSENet, EncoderMixin):
    def __init__(self, num_frames: int, out_channels: int, depth: int = 5, **kwargs):
        super().__init__(**kwargs)

        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        del self.last_linear
        del self.avg_pool

    def get_stages(self):
        return [
            nn.Identity(),
            self.layer0[:-1],
            nn.Sequential(self.layer0[-1], self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
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


TSCSENET_ENCODERS = {
    "tscsenet154": {
        "encoder": TSCSENetEncoder,
        "params": {
            "out_channels": (3, 128, 256, 512, 1024, 2048),
            "block": TSCSEBottleneck,
            "dropout_p": 0.2,
            "groups": 64,
            "layers": [3, 8, 36, 3],
            "num_classes": 1000,
            "reduction": 16,
        },
    },
    "tscse_resnet50": {
        "encoder": TSCSENetEncoder,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": TSCSEResNetBottleneck,
            "layers": [3, 4, 6, 3],
            "downsample_kernel_size": 1,
            "downsample_padding": 0,
            "dropout_p": None,
            "groups": 1,
            "inplanes": 64,
            "input_3x3": False,
            "num_classes": 1000,
            "reduction": 16,
        },
    },
    "se_resnet101": {
        "encoder": TSCSENetEncoder,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": TSCSEResNetBottleneck,
            "layers": [3, 4, 23, 3],
            "downsample_kernel_size": 1,
            "downsample_padding": 0,
            "dropout_p": None,
            "groups": 1,
            "inplanes": 64,
            "input_3x3": False,
            "num_classes": 1000,
            "reduction": 16,
        },
    },
    "se_resnet152": {
        "encoder": TSCSENetEncoder,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": TSCSEResNetBottleneck,
            "layers": [3, 8, 36, 3],
            "downsample_kernel_size": 1,
            "downsample_padding": 0,
            "dropout_p": None,
            "groups": 1,
            "inplanes": 64,
            "input_3x3": False,
            "num_classes": 1000,
            "reduction": 16,
        },
    },
}


def get_encoder(
    name: str,
    num_frames: int,
    in_channels: int = 3,
    depth: int = 5,
    output_stride: _size_3_t = 32,
):
    try:
        encoder = TSCSENET_ENCODERS[name]["encoder"]
    except KeyError as e:
        raise KeyError(
            f"Wrong encoder name `{name}`, supported encoders: {list(TSCSENET_ENCODERS.keys())}"
        ) from e

    params = TSCSENET_ENCODERS[name]["params"]
    params.update(depth=depth)
    params.update(num_frames=num_frames)
    encoder = encoder(**params)

    encoder.set_in_channels(in_channels)
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder
