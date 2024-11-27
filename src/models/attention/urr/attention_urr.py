"""Implementation of uncertain region refinement for residual frames-based U-Net and U-Net++ architectures."""

from __future__ import annotations

# Standard Library
from math import sqrt
from typing import Any, Literal, override

# Third-Party
from einops import rearrange
from segmentation_models_pytorch.base.heads import ClassificationHead, SegmentationHead
from segmentation_models_pytorch.decoders.unet.model import UnetDecoder
from segmentation_models_pytorch.decoders.unetplusplus.model import UnetPlusPlusDecoder

# PyTorch
import torch
from torch import Tensor, nn
from torch.nn import functional as F

# State-of-the-Art (SOTA) code
from thirdparty.AFB_URR.model.AFB_URR import ResBlock

# First party imports
from models.attention.utils import REDUCE_TYPES
from models.two_plus_one import TemporalConvolutionalType
from utils.types import ResidualMode

# Local folders
from ...tscse.tscse import TSCSENetEncoder
from ...tscse.tscse import get_encoder as tscse_get_encoder
from ..model import SpatialAttentionBlock
from ..segmentation_model import ResidualAttentionUnet
from ..utils import get_encoder as smp_get_encoder
from .utils import calc_uncertainty


class UnetDecoderURR(UnetDecoder):
    """U-Net decoder adapted to return upconv decoder layer outputs."""

    @override
    def forward(
        self, *features
    ) -> tuple[
        Tensor, list[Tensor]
    ]:  # pyright: ignore[reportIncompatibleMethodOverride]
        features = features[1:]
        features = features[::-1]

        outputs: list[Tensor] = []

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            outputs.append(x)

        outputs = outputs[::-1]

        return x, outputs


class UnetPlusPlusDecoderURR(UnetPlusPlusDecoder):
    """U-Net++ decoder adapted to return upconv decoder layer outputs."""

    @override
    def forward(self, *features) -> tuple[Tensor, list[Tensor]]:
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](
                        features[depth_idx], features[depth_idx + 1]
                    )
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [
                        dense_x[f"x_{idx}_{dense_l_i}"]
                        for idx in range(depth_idx + 1, dense_l_i + 1)
                    ]
                    cat_features = torch.cat(
                        cat_features + [features[dense_l_i + 1]], dim=1
                    )
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[
                        f"x_{depth_idx}_{dense_l_i}"
                    ](dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features)
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](
            dense_x[f"x_{0}_{self.depth-1}"]
        )

        layer_outputs = [
            dense_x[f"x_{depth_idx}_3"] for depth_idx in range(self.depth - 1, -1, -1)
        ] + [dense_x[f"x_{0}_{self.depth}"]]

        return dense_x[f"x_{0}_{self.depth}"], layer_outputs[::-1]


class RegionRefiner(nn.Module):
    """Region Refinement adapted from AFB-URR."""

    def __init__(
        self, local_size: int, mdim_global: int, mdim_local: int, classes: int
    ):
        """Initialise the RegionRefiner module.

        Args:
            local_size: Size of the pooling layers to correlate local regions.
            mdim_global: Channel dimension of the last layer of the decoder's output.
            mdim_local: Channel dimension to use for the region refinement.
            classes: Number of segmentation classes.

        """
        super().__init__()
        self.local_size = local_size
        self.mdim_global = mdim_global
        self.mdim_local = mdim_local
        self.classes = classes

        self.pred2 = nn.Conv2d(mdim_global, 2, kernel_size=3, padding=1, stride=1)

        self.local_avg = nn.AvgPool2d(local_size, stride=1, padding=local_size // 2)
        self.local_max = nn.MaxPool2d(local_size, stride=1, padding=local_size // 2)
        self.local_conv_fm = nn.Conv2d(
            128, mdim_local, kernel_size=3, padding=1, stride=1
        )
        self.local_res_mm = ResBlock(mdim_local, mdim_local)
        self.local_pred2 = nn.Conv2d(mdim_local, 2, kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    @override
    def forward(
        self, p: Tensor, rough_seg: Tensor, r1: Tensor
    ) -> tuple[Tensor, Tensor]:
        bs, *_ = p.shape
        p = p.unsqueeze(1).expand(-1, self.classes, -1, -1, -1)
        p = rearrange(p, "b k c h w -> (b k) c h w")
        p = self.pred2(F.relu(p))
        p = rearrange(p, "(b k) c h w -> b k c h w", b=bs)
        rough_seg = F.softmax(p, dim=2)[:, :, 1]
        uncertainty = calc_uncertainty(rough_seg).expand(-1, self.classes, -1, -1)
        rough_seg = rearrange(rough_seg.unsqueeze(2), "b k c h w -> (b k) c h w")
        r1 = F.interpolate(r1, scale_factor=2, mode="bilinear", align_corners=False)
        r1 = r1.unsqueeze(1).expand(-1, self.classes, -1, -1, -1)
        r1 = rearrange(r1, "b k c h w -> (b k) c h w")
        r1_weighted = r1 * rough_seg  # (B x K, C, H, W) * (B x K, 1, H, W)

        # Neighbourhood reference.
        r1_local = self.local_avg(r1_weighted)
        r1_local = r1_local / self.local_avg(rough_seg) + 1e-8
        r1_conf = self.local_max(rough_seg)  # (B x K, 1, H, W)

        local_match = torch.cat([r1, r1_local], dim=1)  # (B x K, 2, H, W)
        q = self.local_res_mm(self.local_conv_fm(local_match))
        q = r1_conf * self.local_pred2(F.relu(q))  # (B x K, 2, H, W)

        q = rearrange(q, "(b k) c h w -> b k c h w", b=bs)  # (B, K, C, H, W)

        p = p + uncertainty.unsqueeze(2) * q
        p = F.softmax(p, dim=2)[:, :, 1]

        return p, uncertainty


class URRResidualAttentionUnet(ResidualAttentionUnet):
    """Uncertain region refinement for U-Net with attention mechanism."""

    _default_decoder_channels = [256, 128, 64, 32, 16]
    _default_skip_conn_channels = [2, 5, 10, 20, 40]

    @override
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        residual_mode: ResidualMode = ResidualMode.SUBTRACT_NEXT_FRAME,
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
        temporal_conv_type: TemporalConvolutionalType = TemporalConvolutionalType.TEMPORAL_3D,
        reduce: REDUCE_TYPES = "prod",
        _attention_only: bool = False,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.flat_conv = flat_conv
        self.activation = activation
        self.temporal_conv_type = temporal_conv_type
        self.encoder_name = encoder_name
        self.res_conv_activation = res_conv_activation
        self.reduce: REDUCE_TYPES = reduce
        self.skip_conn_channels = skip_conn_channels
        self.residual_mode = residual_mode
        self._attention_only = _attention_only
        self.classes = classes

        # Define encoder, decoder, segmentation head, and classification head.
        #
        # If `tscse` is a part of the encoder name, handle instantiation slightly
        # differently.
        if "tscse" in encoder_name:
            self.spatial_encoder = tscse_get_encoder(
                encoder_name,
                num_frames=num_frames,
                in_channels=in_channels,
                depth=encoder_depth,
            )

            # NOTE: This is to help with reproducibility during ablation studies.
            with torch.random.fork_rng(devices=("cpu", "cuda:0")):
                self.residual_encoder = tscse_get_encoder(
                    encoder_name,
                    num_frames=num_frames,
                    in_channels=(
                        in_channels
                        if residual_mode == ResidualMode.SUBTRACT_NEXT_FRAME
                        else 2
                    ),
                    depth=encoder_depth,
                )
        else:
            self.spatial_encoder = smp_get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )

            # NOTE: This is to help with reproducibility during ablation studies.
            with torch.random.fork_rng(devices=("cpu", "cuda:0")):
                self.residual_encoder = smp_get_encoder(
                    encoder_name,
                    in_channels=(
                        in_channels
                        if residual_mode == ResidualMode.SUBTRACT_NEXT_FRAME
                        else 2
                    ),
                    depth=encoder_depth,
                    weights=encoder_weights,
                )

        encoder_channels = (
            [x * 2 for x in self.spatial_encoder.out_channels]
            if self.reduce == "cat"
            else self.spatial_encoder.out_channels
        )
        self.decoder = UnetDecoderURR(
            encoder_channels=encoder_channels,
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
                in_channels=self.spatial_encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        # NOTE: Necessary for the SegmentationModel class.
        self.name = f"u-{encoder_name}"

        self.region_refiner = RegionRefiner(7, 16, 32, self.classes)

        self.initialize()

    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, regular_frames: Tensor, residual_frames: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        img_features_list: list[Tensor] = []
        res_features_list: list[Tensor] = []
        b, *_ = regular_frames.shape

        if isinstance(self.encoder, TSCSENetEncoder):
            for img, r_img in zip(regular_frames, residual_frames, strict=False):
                self.check_input_shape(img)
                self.check_input_shape(r_img)

            img_reshaped = rearrange(regular_frames, "b f c h w -> b c f h w")
            res_reshaped = rearrange(residual_frames, "b f c h w -> b c f h w")

            img_features_list = self.spatial_encoder(img_reshaped)
            res_features_list = self.residual_encoder(res_reshaped)

        else:
            for imgs, r_imgs in zip(regular_frames, residual_frames, strict=False):
                self.check_input_shape(imgs)
                self.check_input_shape(r_imgs)

            img_features_list = self.spatial_encoder(
                rearrange(regular_frames, "b f c h w -> (b f) c h w")
            )
            res_features_list = self.spatial_encoder(
                rearrange(regular_frames, "b f c h w -> (b f) c h w")
            )

        residual_outputs: list[Tensor | list[str]] = [["EMPTY"]]

        for i in range(1, 6):
            if isinstance(self.encoder, TSCSENetEncoder):
                img_outputs = rearrange(img_features_list[i], "b c f h w -> b f c h w")
                res_outputs = rearrange(res_features_list[i], "b c f h w -> b f c h w")
            else:
                img_outputs = rearrange(
                    img_features_list[i], "(b f) c h w -> b f c h w", b=b
                )
                res_outputs = rearrange(
                    res_features_list[i], "(b f) c h w -> b f c h w", b=b
                )

            res_block: SpatialAttentionBlock = self.res_layers[
                i - 1
            ]  # pyright: ignore[reportAssignmentType] False positive

            skip_output = res_block(
                st_embeddings=img_outputs, res_embeddings=res_outputs
            )

            if self.reduce == "cat":
                skip_output = rearrange(skip_output, "d b c h w -> b (d c) h w")

            residual_outputs.append(skip_output)

        decoder_output, decoder_outputs = self.decoder(*residual_outputs)
        rough_seg = self.segmentation_head(decoder_output)

        score, initial_uncertainty = self.region_refiner(
            decoder_outputs[0], rough_seg, residual_outputs[1]
        )

        uncertainty: Tensor = calc_uncertainty(F.softmax(score, dim=1))
        uncertainty = uncertainty.view(b, -1).norm(p=2, dim=1) / sqrt(
            regular_frames.shape[-2] * regular_frames.shape[-1]
        )  # [B,1,H,W]
        uncertainty = uncertainty.mean()  # pyright: ignore[reportOptionalMemberAccess]

        score = torch.clamp(score, 1e-7, 1 - 1e-7)
        score = torch.log((score / (1 - score)))

        return score, initial_uncertainty, uncertainty


class URRResidualAttentionUnetPlusPlus(URRResidualAttentionUnet):
    """U-Net++ with attention mechanism and uncertain region refinement."""

    _default_decoder_channels = [256, 128, 64, 32, 16]
    _default_skip_conn_channels = [2, 5, 10, 20, 40]

    @override
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        residual_mode: ResidualMode = ResidualMode.SUBTRACT_NEXT_FRAME,
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
        temporal_conv_type: TemporalConvolutionalType = TemporalConvolutionalType.TEMPORAL_3D,
        reduce: REDUCE_TYPES = "prod",
        _attention_only: bool = False,
    ):
        super(URRResidualAttentionUnet, self).__init__()
        self.num_frames = num_frames
        self.flat_conv = flat_conv
        self.activation = activation
        self.temporal_conv_type = temporal_conv_type
        self.encoder_name = encoder_name
        self.res_conv_activation = res_conv_activation
        self.reduce: REDUCE_TYPES = reduce
        self.skip_conn_channels = skip_conn_channels
        self.residual_mode = residual_mode
        self._attention_only = _attention_only
        self.classes = classes

        # Define encoder, decoder, segmentation head, and classification head.
        #
        # If `tscse` is a part of the encoder name, handle instantiation slightly
        # differently.
        if "tscse" in encoder_name:
            self.spatial_encoder = tscse_get_encoder(
                encoder_name,
                num_frames=num_frames,
                in_channels=in_channels,
                depth=encoder_depth,
            )

            # NOTE: This is to help with reproducibility during ablation studies.
            with torch.random.fork_rng(devices=("cpu", "cuda:0")):
                self.residual_encoder = tscse_get_encoder(
                    encoder_name,
                    num_frames=num_frames,
                    in_channels=(
                        in_channels
                        if residual_mode == ResidualMode.SUBTRACT_NEXT_FRAME
                        else 2
                    ),
                    depth=encoder_depth,
                )
        else:
            self.spatial_encoder = smp_get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )

            # NOTE: This is to help with reproducibility during ablation studies.
            with torch.random.fork_rng(devices=("cpu", "cuda:0")):
                self.residual_encoder = smp_get_encoder(
                    encoder_name,
                    in_channels=(
                        in_channels
                        if residual_mode == ResidualMode.SUBTRACT_NEXT_FRAME
                        else 2
                    ),
                    depth=encoder_depth,
                    weights=encoder_weights,
                )

        encoder_channels = (
            [x * 2 for x in self.spatial_encoder.out_channels]
            if self.reduce == "cat"
            else self.spatial_encoder.out_channels
        )

        self.decoder = UnetPlusPlusDecoderURR(
            encoder_channels=encoder_channels,
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
                in_channels=self.spatial_encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = f"unetplusplus-{encoder_name}"

        self.region_refiner = RegionRefiner(7, 16, 32, self.classes)

        self.initialize()


# NOTE: This is for debugging purposes
if __name__ == "__main__":
    batch_img = torch.randn((2, 10, 1, 224, 224), dtype=torch.float32).cuda()
    batch_res = torch.randn((2, 10, 1, 224, 224), dtype=torch.float32).cuda()

    model: URRResidualAttentionUnetPlusPlus = URRResidualAttentionUnetPlusPlus(
        "resnet50",
        5,
        "imagenet",
        in_channels=1,
        classes=4,
        num_frames=10,
        temporal_conv_type=TemporalConvolutionalType.TEMPORAL_3D,
        reduce="sum",
    ).cuda()  # pyright: ignore[reportAttributeAccessIssue]

    score, init_uncertainty, uncertainty = model.forward(batch_img, batch_res)

    print("score", score.shape, score.min(), score.max())
    print(
        "initial uncertainty",
        init_uncertainty.shape,
        init_uncertainty.min(),
        init_uncertainty.max(),
    )
    print("uncertainty", uncertainty.shape, uncertainty.min(), uncertainty.max())
