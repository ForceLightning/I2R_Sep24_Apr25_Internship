"""Implement FLA-Net with Segmentation Models PyTorch."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Union, override

import torch
from segmentation_models_pytorch.base import ClassificationHead, SegmentationHead
from segmentation_models_pytorch.base.model import SegmentationModel
from segmentation_models_pytorch.decoders.unet.model import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn
from torch.nn import functional as F


class SEBlock(nn.Module):
    """Squeeze-Excitation Block."""

    @override
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(
            in_channels=input_channels,
            out_channels=internal_neurons,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.up = nn.Conv2d(
            in_channels=internal_neurons,
            out_channels=input_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.input_channels = input_channels

    @override
    def forward(self, inputs):
        Gx = torch.norm(inputs, p=2, dim=(2, 3), keepdim=True)  # [1, 1, 1, c]
        x = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class RadixSoftmax(nn.Module):
    """Radix Softmax module."""

    @override
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    @override
    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class FAM(nn.Module):
    """Frequency-based Feature Aggregation Module."""

    @override
    def __init__(self, in_channels):
        super(FAM, self).__init__()
        self.se_block1 = SEBlock(in_channels // 3 * 4, in_channels // 3 * 2)
        self.se_block2 = SEBlock(in_channels // 3 * 4, in_channels // 3 * 2)

        self.reduce = nn.Conv2d(in_channels // 3 * 2, in_channels // 3, 3, 1, padding=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels // 3)

    def shift(self, x, n_segment=3, fold_div=8):
        """Temporally shifts input tensor."""
        z = torch.chunk(x, 3, dim=1)
        x = torch.stack(z, dim=1)
        b, nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(b, n_batch, n_segment, c, h, w)

        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :, :-1, :fold] = x[:, :, 1:, :fold]  # shift left
        out[:, :, 1:, fold : 2 * fold] = x[:, :, :-1, fold : 2 * fold]  # shift right
        out[:, :, :, 2 * fold :] = x[:, :, :, 2 * fold :]  # not shift

        return out.view(b, nt * c, h, w)

    @override
    def forward(self, x):
        _, _, h, w = x.shape
        # temporal shift
        x = self.shift(x)
        # import pdb; pdb.set_trace()
        y = self.upcast(torch.fft.fft2, x)

        y_imag = y.imag
        y_real = y.real

        y1_imag, y2_imag, y3_imag = torch.chunk(y_imag, 3, dim=1)
        y1_real, y2_real, y3_real = torch.chunk(y_real, 3, dim=1)
        # grouping
        pair1 = torch.concat([y1_imag, y2_imag, y1_real, y2_real], dim=1)
        pair2 = torch.concat([y1_imag, y3_imag, y1_real, y3_real], dim=1)

        pair1 = self.se_block1(pair1).float()
        pair2 = self.se_block2(pair2).float()

        y1_real, y1_imag = torch.chunk(pair1, 2, dim=1)
        y1 = torch.complex(y1_real, y1_imag)
        z1 = torch.fft.ifft2(y1, s=(h, w)).float()

        y2_real, y2_imag = torch.chunk(pair2, 2, dim=1)
        y2 = torch.complex(y2_real, y2_imag)
        z2 = torch.fft.ifft2(y2, s=(h, w)).float()

        out = self.reduce(z1 + z2)
        out = F.relu(out)
        out = self.norm(out)

        return out

    def upcast(self, f: Callable[..., Any], x: torch.Tensor):
        """Upcasts an input tensor to float32 for a function call."""
        return f(x.to(torch.float32))


class HeatmapHead(nn.Module):
    """Model head for producing heatmaps."""

    @override
    def __init__(self, input_channels, internal_neurons, out_channels):
        super(HeatmapHead, self).__init__()

        self.upsample1 = nn.ConvTranspose2d(
            input_channels, internal_neurons, 3, stride=2, padding=1
        )
        self.upsample2 = nn.ConvTranspose2d(
            internal_neurons, out_channels, 3, stride=2, padding=1
        )
        self.upsample3 = nn.ConvTranspose2d(out_channels, 1, 3, stride=2, padding=1)

    @override
    def forward(self, inputs):
        outs = []
        if inputs.ndim < 4:
            inputs = inputs.unsqueeze(0)
        w, h = inputs.shape[-2:]
        x = self.upsample1(inputs, output_size=[w * 2, h * 2])
        outs.append(x)
        x = self.upsample2(x, output_size=[w * 4, h * 4])
        outs.append(x)
        x = self.upsample3(x, output_size=[w * 8, h * 8])
        outs.append(x)

        return outs


class FLANetSegmentationModel(SegmentationModel):
    """Segmentation Model wrapper for FLA-Net."""

    @override
    def initialize(self):
        super().initialize()

        fam_list: list[FAM] = []
        heatmaphead_list: list[HeatmapHead] = []
        in_channels = [1024]
        for in_c in in_channels:
            fam_list.append(FAM(3 * in_c))
            heatmaphead_list.append(HeatmapHead(in_c, 256, 64))

        self.fam_list = nn.ModuleList(fam_list)
        self.heatmaphead_list = nn.ModuleList(heatmaphead_list)

    @override
    def forward(self, x: torch.Tensor):
        self.check_input_shape(x)
        fam_outputs = []
        if x.ndim == 5:
            b, f, *_ = x.shape
            x = x.flatten(0, 1)
            feats: list[torch.Tensor] = self.encoder(x)
            features: list[torch.Tensor] = []
            pick_idxs = [4]
            for idx, fea in enumerate(feats):
                c, w, h = fea.shape[1:]
                fea = fea.view(b, f, c, w, h)
                if idx in pick_idxs:
                    pick_idx = idx - pick_idxs[0]
                    tmp_list: list[torch.Tensor] = []
                    tmp_feat_list: list[torch.Tensor] = []
                    for i in range(b):
                        curr_clip_feats = torch.cat(
                            [
                                fea[i][0].unsqueeze(0),
                                fea[i][1].unsqueeze(0),
                                fea[i][2].unsqueeze(0),
                            ],
                            dim=1,
                        )
                        tmp_feat = self.fam_list[pick_idx](curr_clip_feats)
                        y = fea[i][0] + tmp_feat.squeeze()
                        tmp_feat_list.append(tmp_feat)
                        tmp_list.append(y)

                    features.append(torch.stack(tmp_list))
                    fam_outputs.append(torch.stack(tmp_feat_list))
                else:
                    features.append(fea[:, 0, ...])

        else:
            features = self.encoder(x)

        heatmap_decoder: list[torch.Tensor] = []
        for idx, feat in enumerate(fam_outputs):
            heatmap_decoder.append(self.heatmaphead_list[idx](feat.squeeze()))

        decoder_output = self.decoder(*features)

        # Localisation branch.
        heatmap_pred1 = F.interpolate(
            heatmap_decoder[0][-1], size=features[0].shape[-2:], mode="bilinear"
        )
        decoder_output = decoder_output + heatmap_pred1

        masks: torch.Tensor = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels: torch.Tensor = self.classification_head(features[-1])
            return masks, labels

        return masks, [heatmap_pred1]


class Unet(FLANetSegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation.

    Consist of *encoder* and *decoder* parts connected with *skip connections*. Encoder
    extract features of different spatial resolution (skip connections) which are used
    by decoder to define accurate segmentation mask. Use *concatenation* for fusing
    decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder
            (a.k.a backbone) to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage
            generate features two times smaller in spatial dimensions than previous one
            (e.g. for depth 0 we will have features with shapes [(N, C, H, W),], for
            depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"**
            (pre-training on ImageNet) and other pretrained weights (see table with
            available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for
            convolutions used in decoder. Length of the list should be the same as
            **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and
            Activation layers is used. If **"inplace"** InplaceABN will be used, allows
            to decrease memory consumption. Available options are **True, False,
            "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available
            options are **None** and **scse** (https://arxiv.org/abs/1808.08127).
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of
            channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**,
                **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification
            head). Auxiliary output is build on top of encoder if **aux_params** is not
            **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    _default_decoder_channels = [256, 128, 64, 32, 16]

    @override
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = _default_decoder_channels,
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

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
            center=True if encoder_name.startswith("vgg") else False,
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

        self.name = "u-{}".format(encoder_name)
        self.initialize()
