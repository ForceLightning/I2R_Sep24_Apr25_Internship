"""TSCSENet model implementation."""

from __future__ import annotations

import abc
from typing import Any, Callable, Literal, OrderedDict, Type, Union, override

import torch
from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.decoders.unet.model import UnetDecoder
from segmentation_models_pytorch.encoders._base import EncoderMixin
from torch import nn
from torch.nn.common_types import _size_3_t

from models.common import ENCODER_OUTPUT_SHAPES
from models.two_plus_one import DilatedOneD, OneD, compress_2, compress_dilated


class TSCSEModule(nn.Module):
    """Temporal, Spatial Squeeze, and Excitation Module."""

    def __init__(
        self,
        in_depth: int,
        in_channels: int,
        height: int,
        width: int,
        reduction: int = 16,
    ):
        """Initialise the TSCSE module.

        Args:
            in_depth: Number of frames in the input tensor.
            in_channels: Number of input channels.
            height: Height of the input tensor.
            width: Width of the input tensor.
            reduction: Reduction ratio for the squeeze operation
                (default: 16).

        """
        super().__init__()
        self.in_depth = in_depth
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.reduction = reduction
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

        self.sSE = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, height, width)),
            nn.Conv3d(in_channels, 1, 1),
            nn.Sigmoid(),
        )

        self.tSE = nn.Sequential(
            nn.AdaptiveAvgPool3d((in_depth, 1, 1)),
            nn.Conv3d(in_channels, 1, 1),
            nn.Sigmoid(),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cSE = self.cSE(x)
        sSE = self.sSE(x)
        tSE = self.tSE(x)
        return x * cSE + x * sSE + x * tSE


class Bottleneck(nn.Module, abc.ABC):
    """Base class for bottleneck block of TSCSENet."""

    expansion: int

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.layers(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.tscse(out) + residual
        out = self.relu(out)

        return out


class TSCSEBottleneck(Bottleneck):
    """Bottleneck block for SENet-based TSCSENet."""

    expansion: int = 4

    @override
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
        """Initialise the bottleneck block.

        Args:
            in_depth: Number of frames in the input tensor.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            height: Height of the input tensor.
            width: Width of the input tensor.
            groups: Number of groups for the 3x3 convolution.
            reduction: Reduction ratio for the squeeze operation.
            stride: Stride of the first convolution (default: 1).
            downsample: Downsample layer (default: None).

        """
        conv1 = nn.Conv3d(
            in_channels, out_channels * 2, kernel_size=1, bias=False, stride=stride
        )
        bn1 = nn.BatchNorm3d(out_channels * 2)
        conv2 = nn.Conv3d(
            out_channels * 2,
            out_channels * 4,
            kernel_size=(1, 3, 3),
            stride=stride,
            padding=(0, 1, 1),
            groups=groups,
            bias=False,
        )
        bn2 = nn.BatchNorm3d(out_channels * 4)
        conv3 = nn.Conv3d(out_channels * 4, out_channels * 4, kernel_size=1, bias=False)
        bn3 = nn.BatchNorm3d(out_channels * 4)

        self.layers = nn.Sequential(conv1, bn1, conv2, bn2, conv3, bn3)

        self.relu = nn.ReLU(inplace=True)
        self.tscse = TSCSEModule(in_depth, out_channels * 4, height, width, reduction)
        self.downsample = downsample
        self.stride = stride


class TSCSEResNetBottleneck(Bottleneck):
    """Bottleneck block for ResNet-based TSCSENet."""

    expansion: int = 4

    @override
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
        """Initialise the bottleneck block.

        Args:
            in_depth: Number of frames in the input tensor.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            height: Height of the input tensor.
            width: Width of the input tensor.
            groups: Number of groups for the 3x3 convolution.
            reduction: Reduction ratio for the squeeze operation.
            stride: Stride of the first convolution (default: 1).
            downsample: Downsample layer (default: None).

        """
        super().__init__()
        conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, bias=False, stride=stride
        )
        bn1 = nn.BatchNorm3d(out_channels)
        conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=groups,
            bias=False,
        )
        bn2 = nn.BatchNorm3d(out_channels)
        conv3 = nn.Conv3d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        bn3 = nn.BatchNorm3d(out_channels * 4)

        self.layers = nn.Sequential(conv1, bn1, conv2, bn2, conv3, bn3)

        self.relu = nn.ReLU(inplace=True)
        self.tscse = TSCSEModule(in_depth, out_channels * 4, height, width, reduction)
        self.downsample = downsample
        self.stride = stride


class TSCSENet(nn.Module):
    """TSCSENet model."""

    def __init__(
        self,
        name: Literal[
            "tscsenet154", "tscse_resnet50", "tscse_resnet101", "tscse_resnet152"
        ],
        block: type[TSCSEBottleneck] | type[TSCSEResNetBottleneck],
        depth: int,
        layers: list[int],
        num_frames: int,
        groups: int,
        reduction: int,
        dropout_p: float = 0.2,
        inplanes: int = 128,
        input_3x3: bool = True,
        downsample_kernel_size: int | _size_3_t = (1, 3, 3),
        downsample_padding: int | _size_3_t = (0, 1, 1),
        num_classes: int = 1000,
    ):
        """Initialise the TSCSENet model.

        Args:
            name: Name of the model.
            block: Type of the bottleneck block.
            depth: Depth of the model.
            layers: Number of layers in each block.
            num_frames: Length of input sequence.
            groups: Number of groups for the 3x3 convolution.
            reduction: Reduction ratio for the squeeze operation.
            dropout_p: Dropout probability (default: 0.2).
            inplanes: Number of input channels (default: 128).
            input_3x3: Whether to use 3x3 convolutions in the first layer
                (default: True).
            downsample_kernel_size: Kernel size for the downsample convolution
                (default: 3).
            downsample_padding: Padding for the downsample convolution
                (default: 1).
            num_classes: Number of classes (default: 1000).

        """
        super().__init__()
        self.inplanes = inplanes
        self.name: Literal[
            "tscsenet154", "tscse_resnet50", "tscse_resnet101", "tscse_resnet152"
        ] = name
        self.depth = depth
        self._num_frames = num_frames
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
                        padding=(0, 3, 3),
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
            layer_index=1,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0,
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            layer_index=2,
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
            layer_index=3,
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
            layer_index=4,
            blocks=layers[3],
            stride=(1, 2, 2),
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
        )
        self.avg_pool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[TSCSEBottleneck, TSCSEResNetBottleneck]],
        planes: int,
        layer_index: int,
        blocks: int,
        groups: int,
        reduction: int,
        stride: _size_3_t = 1,
        downsample_kernel_size: _size_3_t = 1,
        downsample_padding: _size_3_t = 0,
    ):
        """Create a layer of blocks.

        Args:
            block: Type of the bottleneck block.
            planes: Number of output channels.
            layer_index: Index of the layer.
            blocks: Number of blocks.
            groups: Number of groups for the 3x3 convolution.
            reduction: Reduction ratio for the squeeze operation.
            stride: Stride of the first convolution (default: 1).
            downsample_kernel_size: Kernel size for the downsample convolution
                (default: 1).
            downsample_padding: Padding for the downsample convolution
                (default: 0).

        """
        downsample = None
        if (
            (not isinstance(stride, tuple) and stride != 1)
            or (isinstance(stride, tuple) and stride[1] != 1 and stride[2] != 1)
            or self.inplanes != planes * block.expansion
        ):
            _ds_conv = nn.Conv3d(
                self.inplanes,
                planes * block.expansion,
                kernel_size=downsample_kernel_size,
                stride=stride,
                padding=downsample_padding,
                bias=False,
            )
            downsample = nn.Sequential(
                _ds_conv,
                nn.BatchNorm3d(planes * block.expansion),
            )

        _, height, width = ENCODER_OUTPUT_SHAPES[self.name][layer_index]
        layers = []
        layers.append(
            block(
                self._num_frames,
                self.inplanes,
                planes,
                height=height,
                width=width,
                groups=groups,
                reduction=reduction,
                stride=stride,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self._num_frames,
                    self.inplanes,
                    planes,
                    height,
                    width,
                    groups,
                    reduction,
                )
            )

        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Features extracted from the input tensor.

        """
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the logits from the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Logits computed from the input tensor.

        """
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.logits(x)

        return x


def replace_strides_with_dilation_3d(module: nn.Module, dilation_rate: int):
    """Replace strides with dilation in a 3D convolutional module.

    Args:
        module: 3D convolutional module.
        dilation_rate: Dilation rate.

    """
    for mod in module.modules():
        if isinstance(mod, nn.Conv3d):
            mod.stride = (1, 1, 1)
            mod.dilation = (1, dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = (0, (kh // 2) * dilation_rate, (kw // 2) * dilation_rate)


class Encoder3DMixin(EncoderMixin):
    """Mixin for 3D encoders."""

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
        for stage_indx, dilation_rate in zip(stage_list, dilation_list, strict=True):
            replace_strides_with_dilation_3d(
                module=stages[stage_indx], dilation_rate=dilation_rate
            )

    @property
    def output_stride(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> _size_3_t:
        """Get the output stride of the encoder."""
        if isinstance(self._output_stride, int):
            return min(self._output_stride, 2**self._depth)
        hw = min(max(self._output_stride[1:]), 2**self._depth)
        return (self._output_stride[0], hw, hw)


class TSCSENetEncoder(TSCSENet, Encoder3DMixin):
    """TSCSENet encoder."""

    def __init__(self, out_channels: list[int], depth: int = 5, **kwargs):
        """Initialise the TSCSENet encoder.

        Args:
            num_frames: Number of frames in the input tensor.
            out_channels: Number of output channels.
            depth: Depth of the model (default: 5).
            kwargs: Additional arguments.

        """
        kwargs |= {"depth": depth}
        super().__init__(**kwargs)

        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        del self.last_linear
        del self.avg_pool

    @override
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

    @override
    def set_in_channels(self, in_channels: int, pretrained: bool = True):
        """Change first convolution channels."""
        if in_channels == 3:
            return
        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        self._patch_first_conv(new_in_channels=in_channels, pretrained=pretrained)

    def _patch_first_conv(
        self,
        new_in_channels: int,
        default_in_channels: int = 3,
        pretrained: bool = True,
    ) -> None:

        # Get first conv
        module = None
        for module in self.modules():
            if (
                isinstance(module, nn.Conv3d)
                and module.in_channels == default_in_channels
            ):
                break

        if isinstance(module, nn.Conv3d):
            weight = module.weight.detach()
            module.in_channels = new_in_channels
            if not pretrained:
                module.weight = nn.Parameter(
                    torch.Tensor(
                        module.out_channels,
                        new_in_channels // module.groups,
                        *module.kernel_size,
                    )
                )
                module.reset_parameters()
            elif new_in_channels == 1:
                new_weight = weight.sum(1, keepdim=True)
                module.weight = nn.Parameter(new_weight)
            else:
                new_weight = torch.Tensor(
                    module.out_channels,
                    new_in_channels // module.groups,
                    *module.kernel_size,
                )

                for i in range(new_in_channels):
                    new_weight[:, i] = weight[:, i % default_in_channels]

                new_weight = new_weight * (default_in_channels / new_in_channels)
                module.weight = nn.Parameter(new_weight)


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
) -> TSCSENetEncoder:
    """Get a TSCSENet encoder by name.

    Args:
        name: Name of the encoder.
        num_frames: Number of frames in the input tensor.
        in_channels: Number of input channels (default: 3).
        depth: Depth of the model (default: 5).
        output_stride: Output stride (default: 32).

    Returns:
        TSCSENet encoder.

    """
    try:
        encoder: Type[TSCSENetEncoder] = TSCSENET_ENCODERS[name]["encoder"]
    except KeyError as e:
        raise KeyError(
            f"Wrong encoder name `{name}`, supported encoders: {list(TSCSENET_ENCODERS.keys())}"
        ) from e

    params = TSCSENET_ENCODERS[name]["params"]
    params |= {
        "num_frames": num_frames,
        "depth": depth,
        "name": name,
    }
    encoder_obj = encoder(**params)

    encoder_obj.set_in_channels(in_channels, pretrained=False)
    if output_stride != 32:
        encoder_obj.make_dilated(output_stride)

    return encoder_obj


class TSCSEUnet(SegmentationModel):
    """Temporal, Spatial Squeeze, and Channel Excitation U-Net."""

    _default_decoder_channels = [256, 128, 64, 32, 16]
    _default_skip_conn_channels = [2, 5, 10, 20, 40]

    def __init__(
        self,
        encoder_name: str = "tscse_resnet50",
        encoder_depth: int = 5,
        decoder_use_batchnorm: bool = True,
        decoder_channels: list[int] = _default_decoder_channels,
        decoder_attention_type: Literal["scse"] | None = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: str | Type[nn.Module] | None = None,
        skip_conn_channels: list[int] = _default_skip_conn_channels,
        num_frames: Literal[5, 10, 15, 20, 30] = 5,
        aux_params: dict[str, Any] | None = None,
        flat_conv: bool = False,
        res_conv_activation: str | None = None,
        use_dilations: bool = False,
    ) -> None:
        """Initialise the TSCSE-U-Net model.

        Args:
            encoder_name: Name of the encoder.
            encoder_depth: Depth of the encoder.
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

        # Define encoder, decoder, segmentation head, and classification head.
        self.encoder = get_encoder(
            encoder_name,
            num_frames=num_frames,
            in_channels=in_channels,
            depth=encoder_depth,
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
            kernel_size=3,
            activation=activation,
        )

        self.classification_head = (
            ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
            if aux_params is not None
            else None
        )

        self.name = f"u-{encoder_name}"
        self.compress: Callable[..., torch.Tensor]
        self.initialize()

    @override
    def initialize(self):
        super().initialize()

        # INFO: Parameter tuning for output channels
        onedlayers: list[OneD | DilatedOneD] = []
        for i, out_channels in enumerate(self.skip_conn_channels):
            mod: OneD | DilatedOneD
            if self.use_dilations and self.num_frames in [5, 30]:
                _channels, h, w = ENCODER_OUTPUT_SHAPES[self.encoder_name][i]
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
    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if isinstance(output_stride, tuple):
            hs, ws = output_stride[1:]
        else:
            hs = ws = output_stride

        if h % hs != 0 or w % ws != 0:
            new_h = (h // hs + 1) * hs if h % hs != 0 else h
            new_w = (w // ws + 1) * ws if w % ws != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {(hs, ws)}. Consider pad your images to shape ({new_h}, {new_w})."
            )

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

        for img in x:
            self.check_input_shape(img)

        x_reshaped = x.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W)
        features_list = self.encoder(x_reshaped)  # Output: (6, B, C, F, H, W)

        # Goes through each layer and gets the output from that layer from all the
        # feature outputs.
        # PERF: Maybe this can be done in parallel?
        for index in range(1, 6):
            layer_output = features_list[index].permute(
                0, 2, 1, 3, 4
            )  # (B, F, C, H, W)

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
