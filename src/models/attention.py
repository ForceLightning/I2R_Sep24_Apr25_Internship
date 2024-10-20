# -*- coding: utf-8 -*-
"""U-Net with Attention mechanism on residual frames."""

from __future__ import annotations

from typing import Any, Literal, override

import torch
from segmentation_models_pytorch.base.heads import ClassificationHead, SegmentationHead
from segmentation_models_pytorch.base.model import SegmentationModel
from segmentation_models_pytorch.decoders.unet.model import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder as smp_get_encoder
from torch import nn
from torch.nn import functional as F

from models.common import ENCODER_OUTPUT_SHAPES
from models.tscse import get_encoder as tscse_get_encoder
from models.two_plus_one import DilatedOneD, OneD, compress_2, compress_dilated
from utils.utils import ResidualMode

REDUCE_TYPES = Literal["sum", "prod", "cat", "weighted", "weighted_learnable"]


class AttentionLayer(nn.Module):
    """Attention mechanism between spatio-temporal and spatial embeddings.

    As the spatial dimensions of the image can be considered the sequence to be
    processed, the channel dimension must be the embedding dimension for each part
    of Q, K, and V tensors.
    """

    def __init__(
        self,
        embed_dim: int,
        num_frames: int,
        num_heads: int = 1,
        key_embed_dim: int | None = None,
        value_embed_dim: int | None = None,
        need_weights: bool = False,
        reduce: REDUCE_TYPES = "sum",
    ) -> None:
        """Initialise the attention mechanism.

        As the spatial dimensions of the image can be considered the sequence to be
        processed, the channel dimension must be the embedding dimension for each part
        of Q, K, and V tensors.

        Args:
            embed_dim: The number of expected features in the input.
            num_frames: The number of frames in the sequence.
            num_heads: Number of attention heads.
            key_embed_dim: The dimension of the key embeddings. If None, it is the same
                as the embedding dimension.
            value_embed_dim: The dimension of the value embeddings. If None, it is the
                same as the embedding dimension.
            need_weights: Whether to return the attention weights. (Not functional).
            reduce: How to reduce the attention outputs.

        """
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = key_embed_dim if key_embed_dim else embed_dim
        self.vdim = value_embed_dim if key_embed_dim else embed_dim
        self.num_heads = num_heads
        self.need_weights = need_weights
        self.num_frames = num_frames
        self.reduce: REDUCE_TYPES = reduce

        # Create a MultiheadAttention module for each frame in the sequence.
        attentions = []
        for _ in range(self.num_frames):
            mha = nn.MultiheadAttention(
                self.embed_dim,
                self.num_heads,
                kdim=self.kdim,
                vdim=self.vdim,
                batch_first=False,
            )
            attentions.append(mha)

        self.attentions = nn.ModuleList(attentions)

    @override
    def forward(
        self, q: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor
    ) -> torch.Tensor:
        # Get the dimensions of the input tensors.
        if ks.ndim == 4:
            q = q.view(1, *q.shape)
            ks = ks.view(1, *ks.shape)
            vs = vs.view(1, *vs.shape)
        b, f, c, h, w = ks.shape

        # Reshape the input tensors to the expected shape for the MultiheadAttention
        # module.
        # Q: (<B>, H, W, C) -> (H, W, <B>, C) -> (H * W, <B>, C) [L, <N>, E_q]
        # K: (<B>, F, H, W, C) -> (F, H, W, <B>, C) -> (F, H * W, <B>, C) [S, <N>, E_k]
        # V: (<B>, F, H, W, C) -> (F, H, W, <B>, C) -> (F, H * W, <B>, C) [S, <N>, E_v]
        # NOTE: For K and V, we iterate over the frames in the sequence.
        q_vec = q.flatten(2, 3).permute(2, 0, 1)  # (B, C, H * W) -> (H * W, B, C)
        k_vec = ks.flatten(3, 4).permute(  # (B, F, C, H * W)
            1, 3, 0, 2
        )  # (F, H * W, B, C)
        v_vec = vs.flatten(3, 4).permute(  # (B, F, C, H * W)
            1, 3, 0, 2
        )  # (F, H * W, B, C)

        attn_outputs: list[torch.Tensor] = []
        for i in range(f):  # Iterate over the frames in the sequence.
            # Input to attention is:
            # Q: (H * W, B, C) or (H * W, C)
            # K: (H * W, B, C) or (H * W, C)
            # V: (H * W, B, C) or (H * W, C)
            # INFO: Maybe we should return the weights? Not sure.
            out, _weights = self.attentions[i](
                q_vec, k_vec[i], v_vec[i], need_weights=self.need_weights
            )
            attn_outputs.append(out)

        # NOTE: Maybe don't sum this here, if we want to do weighted averages.
        match self.reduce:
            case "sum" | "cat" | "prod":
                attn_output_t = (
                    torch.stack(attn_outputs, dim=0)  # (F, H * W, B, C)
                    .sum(dim=0)  # (H * W, B, C)
                    .view(h, w, b, c)
                    .permute(2, 3, 0, 1)  # (B, C, H, W)
                )
            case "weighted" | "weighted_learnable":
                attn_output_t = (
                    torch.stack(attn_outputs, dim=0)  # (F, H * W, B, C)
                    .view(f, h, w, b, c)
                    .permute(0, 3, 4, 1, 2)  # (F, B, C, H, W)
                )

        return attn_output_t


class WeightedAverage(nn.Module):
    """A weighted average module."""

    def __init__(self, in_features: int, learnable: bool = False):
        """Initialise the weighted average module.

        Args:
            in_features: Number of features to reduce.
            learnable: Whether the parameter should store gradients.

        """
        super().__init__()
        self.in_features = in_features
        self.learnable = learnable
        if learnable:
            self.weights = nn.Parameter(torch.randn((in_features)), requires_grad=True)
        else:
            self.weights = nn.Parameter(
                torch.tensor(
                    [0.5] + [0.5 / (in_features - 1)] * (in_features - 1),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.in_features, (
            f"Input of shape {x.shape} does not have last dimension "
            + f"matching in_features {self.in_features}"
        )
        if self.learnable:
            weighted_values = F.softmax(self.weights, dim=-1)
        else:
            weighted_values = self.weights
        return F.linear(x, weighted_values)


class SpatialAttentionBlock(nn.Module):
    """Residual block with attention mechanism between spatio-temporal embeddings."""

    def __init__(
        self,
        temporal_conv: OneD | DilatedOneD,
        attention: AttentionLayer,
        num_frames: int,
        reduce: REDUCE_TYPES,
        reduce_dim: int = 0,
        _attention_only: bool = False,
    ):
        """Initialise the residual block.

        Args:
            temporal_conv: Temporal convolutional layer to compress the spatial
            embeddings.
            attention: Attention mechanism between the embeddings.
            num_frames: Number of frames per input.
            reduce: The reduction method to apply to the concatenated embeddings.
            reduce_dim: The dimension to reduce the concatenated embeddings.

        """
        super().__init__()
        self.temporal_conv = temporal_conv
        self.attention = attention
        self._attention_only = _attention_only
        self._reduce = reduce
        match reduce:
            case "cat":
                self.reduce = nn.Identity()
            case "weighted":
                self.reduce = WeightedAverage(num_frames + 1, False)
            case "weighted_learnable":
                self.reduce = WeightedAverage(num_frames + 1, True)

    def forward(self, st_embeddings: torch.Tensor, res_embeddings: torch.Tensor):
        """Forward pass of the residual block.

        Args:
            st_embeddings: Spatio-temporal embeddings from raw frames.
            res_embeddings: Spatial embeddings from residual frames.

        """
        # Output is of shape (B, C, H, W)
        if isinstance(self.temporal_conv, OneD):
            compress_output = compress_2(st_embeddings, self.temporal_conv)
        else:
            compress_output = compress_dilated(st_embeddings, self.temporal_conv)

        attention_output: torch.Tensor = self.attention(
            q=compress_output, ks=res_embeddings, vs=res_embeddings
        )

        # NOTE: This is for debugging purposes only.
        if self._attention_only:
            return attention_output

        b, c, h, w = compress_output.shape
        if self._reduce == "prod":
            res = torch.mul(compress_output, attention_output)
            return res

        compress_output = compress_output.view(1, b, c, h, w)
        attention_output = attention_output.view(-1, b, c, h, w)
        out = torch.cat((compress_output, attention_output), dim=0)
        if self._reduce == "sum":
            return torch.sum(input=out, dim=0).view(b, c, h, w)
        return self.reduce(out)


class ResidualAttentionUnet(SegmentationModel):
    """U-Net with Attention mechanism on residual frames."""

    _default_decoder_channels = [256, 128, 64, 32, 16]
    _default_skip_conn_channels = [2, 5, 10, 20, 40]

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
        use_dilations: bool = False,
        reduce: REDUCE_TYPES = "prod",
        _attention_only: bool = False,
    ):
        """Initialise the model.

        Args:
            encoder_name: Name of the encoder.
            encoder_depth: Depth of the encoder.
            encoder_weights: Weights of the encoder.
            residual_mode: Mode of the residual frames calculation.
            decoder_use_batchnorm: Whether to use batch normalization in the decoder.
            decoder_channels: Number of channels in the decoder.
            decoder_attention_type: Type of attention in the decoder.
            in_channels: Number of channels in the input image.
            classes: Number of classes in the output mask.
            activation: Activation function to use.
            skip_conn_channels: Number of channels in each skip connection's temporal
                convolutions.
            num_frames: Number of frames in the sequence.
            aux_params: Auxiliary parameters for the classification head.
            flat_conv: Whether to use flat convolutions.
            res_conv_activation: Activation function to use in the residual
                convolutions.
            use_dilations: Whether to use dilated conv.
            reduce: How to reduce the post-attention features and the original features.
            _attention_only: Whether to return only the attention output.

        """
        super().__init__()
        self.num_frames = num_frames
        self.flat_conv = flat_conv
        self.activation = activation
        self.use_dilations = use_dilations
        self.encoder_name = encoder_name
        self.res_conv_activation = res_conv_activation
        self.reduce: REDUCE_TYPES = reduce
        self.skip_conn_channels = skip_conn_channels
        self.residual_mode = residual_mode
        self._attention_only = _attention_only

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
        self.decoder = UnetDecoder(
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
        self.initialize()

    @override
    def initialize(self) -> None:
        super().initialize()

        # Residual connection layers.
        res_layers: list[nn.Module] = []
        for i, out_channels in enumerate(self.skip_conn_channels):
            # (1): Create the 1D temporal convolutional layer.
            oned: OneD | DilatedOneD
            c, h, w = ENCODER_OUTPUT_SHAPES[self.encoder_name][i]
            if self.use_dilations and self.num_frames in [5, 30]:
                oned = DilatedOneD(
                    1,
                    out_channels,
                    self.num_frames,
                    h * w,
                    flat=self.flat_conv,
                    activation=self.res_conv_activation,
                )

            else:
                oned = OneD(
                    1,
                    out_channels,
                    self.num_frames,
                    self.flat_conv,
                    self.res_conv_activation,
                )

            # (2): Create the attention mechanism.
            # NOTE: This is to help with reproducibility during ablation studies.
            with torch.random.fork_rng(devices=("cpu", "cuda:0")):
                attention = AttentionLayer(
                    c, num_heads=1, num_frames=self.num_frames, need_weights=False
                )

                res_block = SpatialAttentionBlock(
                    oned,
                    attention,
                    num_frames=self.num_frames,
                    reduce=self.reduce,
                    _attention_only=self._attention_only,
                )
                res_layers.append(res_block)

        self.res_layers = nn.ModuleList(res_layers)

    @property
    def encoder(self):
        """Get the encoder of the model."""
        # NOTE: Necessary for the decoder.
        return self.spatial_encoder

    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, regular_frames: torch.Tensor, residual_frames: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            regular_frames: Regular frames from the sequence.
            residual_frames: Residual frames from the sequence.

        """
        # Output features by batch and then by encoder layer.
        img_features_list: list[list[torch.Tensor]] = []
        res_features_list: list[list[torch.Tensor]] = []

        # Go through by batch and get the results for each layer of the encoder.
        for imgs, r_imgs in zip(regular_frames, residual_frames, strict=False):
            self.check_input_shape(imgs)
            self.check_input_shape(r_imgs)

            img_features = self.spatial_encoder(imgs)
            img_features_list.append(img_features)
            res_features = self.residual_encoder(r_imgs)
            res_features_list.append(res_features)

        residual_outputs: list[torch.Tensor | list[str]] = [["EMPTY"]]

        for i in range(1, 6):
            # Now inputs to the attn block are stacked by batch dimension first.
            img_outputs = torch.stack([outputs[i] for outputs in img_features_list])
            res_outputs = torch.stack([outputs[i] for outputs in res_features_list])

            res_block: SpatialAttentionBlock = self.res_layers[
                i - 1
            ]  # pyright: ignore[reportAssignmentType] False positive

            skip_output = res_block(
                st_embeddings=img_outputs, res_embeddings=res_outputs
            )

            if self.reduce == "cat":
                d, b, c, h, w = skip_output.shape
                skip_output = skip_output.permute(1, 0, 2, 3, 4).reshape(b, d * c, h, w)

            residual_outputs.append(skip_output)

        decoder_output = self.decoder(*residual_outputs)

        masks = self.segmentation_head(decoder_output)
        return masks

    @override
    @torch.no_grad()
    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, regular_frames: torch.Tensor, residual_frames: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            self.eval()

        x = self.forward(regular_frames, residual_frames)
        return x
