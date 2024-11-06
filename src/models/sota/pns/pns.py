from typing import Sequence, override

import torch
from thirdparty.VPS.lib.module.LightRFB import LightRFB
from thirdparty.VPS.lib.module.PNSPlusModule import (
    Relevance_Measuring,
    Spatial_Temporal_Aggregation,
)
from thirdparty.VPS.lib.module.PNSPlusNetwork import PNSNet as PNSBaseNet
from thirdparty.VPS.lib.module.PNSPlusNetwork import conbine_feature as CombineFeatures
from thirdparty.VPS.lib.module.PNSPlusNetwork import res2net50_v1b_26w_4s
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t


class NSBlock(nn.Module):
    """Normalised Self-Attention block from the paper.

    Adapted to have variable shape inputs.
    """

    def __init__(
        self,
        image_dims: _size_2_t,
        in_channels: int = 32,
        n_head: int = 4,
        d_k: int = 8,
        d_v: int = 8,
        radius: Sequence[int] = (3, 3, 3, 3),
        dilation: Sequence[int] = (1, 3, 5, 7),
    ):
        """Initialise the Normalised Self-Attention block.

        Args:
            image_dims: (height, width) or height if height == width.
            in_channels: Number of input channels.
            n_head: Number of attention heads.
            d_k: Total number of features for the key.
            d_v: Total number of features for the value.
            radius: Radius for relevance measuring and spatial-temporal aggregation for
                each head.
            dilation: Dilation for relevance measuring and spatial-temporal aggregation
                for each head.

        """
        super().__init__()
        self.image_dims = image_dims
        self.in_channels = in_channels
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.radius = radius
        self.dilation = dilation
        self.query_conv = nn.Conv3d(in_channels, n_head * d_k, 1, bias=False)
        self.key_conv = nn.Conv3d(in_channels, n_head * d_k, 1, bias=False)
        self.value_conv = nn.Conv3d(in_channels, n_head * d_v, 1, bias=False)
        self.output_linear = nn.Conv3d(n_head * d_v, in_channels, 1, bias=False)
        # OPTIM: Self-adapting layer norm
        norm_shape = [
            int(self.in_channels / self.n_head),
            (
                self.image_dims[0]
                if isinstance(self.image_dims, tuple)
                else self.image_dims
            ),
            (
                self.image_dims[1]
                if isinstance(self.image_dims, tuple)
                else self.image_dims
            ),
        ]
        self.bn = nn.LayerNorm(norm_shape)

    @override
    def forward(self, first: Tensor, x: Tensor) -> Tensor:
        dilation, radius = self.dilation, self.radius
        x_ = x.permute(0, 2, 1, 3, 4).contiguous()
        first_ = first.permute(0, 2, 1, 3, 4).contiguous()
        query = self.query_conv(first_).permute(0, 2, 1, 3, 4)
        query_chunk = query.chunk(self.n_head, 2)
        key = self.key_conv(x_).permute(0, 2, 1, 3, 4)
        key_chunk = key.chunk(self.n_head, 2)
        value = self.value_conv(x_).permute(0, 2, 1, 3, 4)
        value_chunk = value.chunk(self.n_head, 2)

        M_T: list[Tensor] = []
        M_A: list[Tensor] = []

        for i in range(self.n_head):
            query_i = query_chunk[i].contiguous()
            query_i = self.bn(query_i)
            key_i = key_chunk[i].contiguous()
            value_i = value_chunk[i].contiguous()
            # OPTIM: Self-adapting scaling factor
            M_A_i = Relevance_Measuring.apply(
                query_i, key_i, radius[i], dilation[i]
            ) / ((self.in_channels / self.n_head) ** 0.5)
            M_A.append(F.softmax(M_A_i, dim=2))
            M_T.append(
                Spatial_Temporal_Aggregation.apply(  # pyright: ignore[reportArgumentType]
                    M_A_i, value_i, radius[i], dilation[i]
                )
            )

        M_S, _ = torch.max(torch.cat(M_A, dim=2), dim=2)
        M_T_out: Tensor = torch.cat(M_T, dim=2).permute(0, 2, 1, 3, 4)
        out_cat = self.output_linear(M_T_out) * M_S.unsqueeze(2).permute(0, 2, 1, 3, 4)

        return out_cat.permute(0, 2, 1, 3, 4)


class PNSNet(PNSBaseNet):
    """Implementation of PNS+ Network from doi:10.48550/arXiv.2203.14291.

    This adapts the implementation provided from the repository with variable input
    shapes.

    """

    def __init__(
        self, image_shape: _size_2_t = (256, 448), classes: int = 1, num_frames: int = 6
    ):
        """Initialise the PNS+ model.

        Args:
            image_shape: Resolution of the input video in (height, width).
            classes: Number of output classes.
            num_frames: Number of input frames, which includes the anchor frame.

        """
        super(PNSNet, self).__init__()
        h = image_shape[0] if isinstance(image_shape, (tuple, list)) else image_shape
        w = image_shape[1] if isinstance(image_shape, (tuple, list)) else image_shape

        self.feature_extractor = res2net50_v1b_26w_4s(pretrained=True)
        self.High_RFB = LightRFB()
        self.Low_RFB = LightRFB(channels_in=512, channels_mid=128, channels_out=24)

        self.squeeze = nn.Sequential(
            nn.Conv2d(1024, 32, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.decoder = CombineFeatures()
        self.SegNIN = nn.Sequential(
            nn.Dropout2d(0.1), nn.Conv2d(16, classes, kernel_size=1, bias=False)
        )

        self.NSB_global = NSBlock(
            (h // 16, w // 16), 32, radius=[3, 3, 3, 3], dilation=[3, 4, 3, 4]
        )
        self.NSB_local = NSBlock(
            (h // 16, w // 16), 32, radius=[3, 3, 3, 3], dilation=[1, 2, 1, 2]
        )
        self.num_frames = num_frames

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Tensor of shape (B, F, C, H, W)

        Returns:
            Tensor: Tensor of shape (B, F - 1, num_classes, H, W)

        """
        origin_shape = x.shape
        bs = x.shape[0]
        x = x.view(-1, *origin_shape[2:])
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)
        x1 = self.feature_extractor.layer1(x)

        # Extract anchor, low-level, and high-level features.
        low_features = self.feature_extractor.layer2(x1)
        high_feature = self.feature_extractor.layer3(low_features)

        # Reduce the channel dimension.
        high_feature = self.High_RFB(high_feature)
        low_features = self.Low_RFB(low_features)

        # Reshape into temporal formation.
        high_feature = high_feature.view(*origin_shape[:2], *high_feature.shape[1:])
        low_features = low_features.view(*origin_shape[:2], *low_features.shape[1:])

        # Feature Separation.
        high_feature_global = (
            high_feature[:, 0, ...]
            .unsqueeze(dim=1)
            .repeat(1, (self.num_frames - 1), 1, 1, 1)
        )
        high_feature_local = high_feature[:, 1 : self.num_frames, ...]
        low_feature = low_features[:, 1 : self.num_frames, ...]

        # First NS Block.
        high_feature_1 = (
            self.NSB_global(high_feature_global, high_feature_local)
            + high_feature_local
        )
        # Second NS Block.
        high_feature_2 = self.NSB_local(high_feature_1, high_feature_1) + high_feature_1

        # Residual Connection.
        high_feature = high_feature_2 + high_feature_local

        # Reshape back into spatial formation.
        high_feature = high_feature.contiguous().view(-1, *high_feature.shape[2:])
        low_feature = low_feature.contiguous().view(-1, *low_feature.shape[2:])

        # Resize high-level feature to the same as low-level feature.
        high_feature = F.interpolate(
            high_feature,
            size=(low_feature.shape[-2], low_feature.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        # UNet-like decoder.
        out = self.decoder(low_feature.clone(), high_feature.clone())
        out = torch.sigmoid(
            F.interpolate(
                self.SegNIN(out),
                size=(origin_shape[-2], origin_shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
        )

        out = out.reshape(bs, out.shape[0] // bs, *out.shape[1:])

        return out


if __name__ == "__main__":
    image_dims = (224, 224)
    a = torch.randn(2, 10, 3, *image_dims).cuda()
    model = PNSNet(image_dims, 4, 10).cuda()
    print(model(a).shape)
