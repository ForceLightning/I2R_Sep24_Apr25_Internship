# -*- coding: utf-8 -*-
"""U-Net with Attention mechanism on residual frames."""

from __future__ import annotations

from typing import Any, Literal, OrderedDict, override

import segmentation_models_pytorch as smp
import torch
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from segmentation_models_pytorch.base.heads import ClassificationHead, SegmentationHead
from segmentation_models_pytorch.base.model import SegmentationModel
from segmentation_models_pytorch.decoders.unet.model import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder as smp_get_encoder
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Metric, MetricCollection
from torchvision.transforms.v2 import Compose
from torchvision.utils import draw_segmentation_masks

from metrics.dice import GeneralizedDiceScoreVariant
from metrics.logging import (
    setup_metrics,
    shared_metric_calculation,
    shared_metric_logging_epoch_end,
)
from models.common import ENCODER_OUTPUT_SHAPES, CommonModelMixin
from models.tscse import TSCSENetEncoder
from models.tscse import get_encoder as tscse_get_encoder
from models.two_plus_one import DilatedOneD, OneD, compress_2, compress_dilated
from utils import utils
from utils.types import (
    INV_NORM_GREYSCALE_DEFAULT,
    INV_NORM_RGB_DEFAULT,
    ClassificationMode,
    LoadingMode,
    ResidualMode,
)

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
    def check_input_shape(self, x):
        if isinstance(self.encoder, TSCSENetEncoder):
            self._check_input_shape_tscse(x)
        else:
            super().check_input_shape(x)

    def _check_input_shape_tscse(self, x):
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
        img_features_list: list[torch.Tensor] = []
        res_features_list: list[torch.Tensor] = []

        if isinstance(self.encoder, TSCSENetEncoder):
            for imgs, r_imgs in zip(regular_frames, residual_frames, strict=False):
                self.check_input_shape(imgs)
                self.check_input_shape(r_imgs)

            # (B, F, C, H, W) -> (B, C, F, H, W)
            img_reshaped = regular_frames.permute(0, 2, 1, 3, 4)
            res_reshaped = residual_frames.permute(0, 2, 1, 3, 4)

            img_features_list = self.spatial_encoder(img_reshaped)
            res_features_list = self.residual_encoder(res_reshaped)
        else:
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
            if isinstance(self.encoder, TSCSENetEncoder):
                # (B, C, F, H, W) -> (B, F, C, H, W)
                img_outputs = img_features_list[i].permute(0, 2, 1, 3, 4)
                res_outputs = res_features_list[i].permute(0, 2, 1, 3, 4)
            else:
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


class ResidualAttentionUnetLightning(CommonModelMixin):
    """Attention mechanism-based U-Net."""

    def __init__(
        self,
        batch_size: int,
        metric: Metric | None = None,
        loss: nn.Module | str | None = None,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        num_frames: int = 5,
        weights_from_ckpt_path: str | None = None,
        optimizer: Optimizer | str = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: LRScheduler | str = "gradual_warmup_scheduler",
        scheduler_kwargs: dict[str, Any] | None = None,
        multiplier: int = 2,
        total_epochs: int = 50,
        alpha: float = 1.0,
        _beta: float = 0.0,
        learning_rate: float = 1e-4,
        dl_classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        eval_classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        residual_mode: ResidualMode = ResidualMode.SUBTRACT_NEXT_FRAME,
        loading_mode: LoadingMode = LoadingMode.RGB,
        dump_memory_snapshot: bool = False,
        flat_conv: bool = False,
        unet_activation: str | None = None,
        attention_reduction: REDUCE_TYPES = "sum",
        attention_only: bool = False,
    ):
        """Initialise the Attention mechanism-based U-Net.

        Args:
            batch_size: Mini-batch size.
            metric: Metric to use for evaluation.
            loss: Loss function to use for training.
            encoder_name: Name of the encoder.
            encoder_depth: Depth of the encoder.
            encoder_weights: Weights to use for the encoder.
            in_channels: Number of input channels.
            classes: Number of classes.
            num_frames: Number of frames to use.
            weights_from_ckpt_path: Path to checkpoint file.
            optimizer: Optimizer to use.
            optimizer_kwargs: Optimizer keyword arguments.
            scheduler: Learning rate scheduler to use.
            scheduler_kwargs: Scheduler keyword arguments.
            multiplier: Multiplier for the model.
            total_epochs: Total number of epochs.
            alpha: Weight for the loss.
            _beta: (Unused) Weight for the loss.
            learning_rate: Learning rate.
            dl_classification_mode: Classification mode for the dataloader.
            eval_classification_mode: Classification mode for evaluation.
            residual_mode: Residual calculation mode.
            loading_mode: Loading mode for the images.
            dump_memory_snapshot: Whether to dump memory snapshot.
            flat_conv: Whether to use flat convolutions.
            unet_activation: Activation function for the U-Net.
            attention_reduction: Attention reduction type.
            attention_only: Whether to use attention only.

        """
        super().__init__()
        self.save_hyperparameters(ignore=["metric", "loss"])
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.classes = classes
        self.num_frames = num_frames
        self.dump_memory_snapshot = dump_memory_snapshot

        # Trace memory usage
        if self.dump_memory_snapshot:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python"
            )
        # PERF: The model can be `torch.compile()`'d but layout issues occur with
        # convolutional networks. See: https://github.com/pytorch/pytorch/issues/126585
        self.model = ResidualAttentionUnet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            residual_mode=residual_mode,
            in_channels=in_channels,
            classes=classes,
            num_frames=num_frames,
            flat_conv=flat_conv,
            activation=unet_activation,
            use_dilations=True,
            reduce=attention_reduction,
            _attention_only=attention_only,
        )
        self.residual_mode = residual_mode
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}
        self.loading_mode = loading_mode

        # Sets loss if it's a string
        if isinstance(loss, str):
            match loss:
                case "cross_entropy":
                    class_weights = torch.Tensor(
                        [
                            0.00018531001957368073,
                            0.015518576429048081,
                            0.058786240529692384,
                            0.925509873021686,
                        ],
                    ).to(self.device.type)
                    self.loss = nn.CrossEntropyLoss(weight=class_weights)
                case "focal":
                    self.loss = FocalLoss("multiclass", normalized=True)
                case _:
                    raise NotImplementedError(
                        f"Loss type of {loss} is not implemented!"
                    )
        # Otherwise, set if nn.Module
        else:
            self.loss = (
                loss
                if isinstance(loss, nn.Module)
                # If none
                else (
                    DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
                    if dl_classification_mode == ClassificationMode.MULTILABEL_MODE
                    else DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
                )
            )

        self.multiplier = multiplier
        self.total_epochs = total_epochs
        self.alpha = alpha
        self.de_transform = Compose(
            [
                (
                    INV_NORM_RGB_DEFAULT
                    if loading_mode == LoadingMode.RGB
                    else INV_NORM_GREYSCALE_DEFAULT
                )
            ]
        )
        # NOTE: This is to help with reproducibility
        with torch.random.fork_rng(devices=("cpu", "cuda:0")):
            self.example_input_array = (
                torch.randn(
                    (self.batch_size, self.num_frames, self.in_channels, 224, 224),
                    dtype=torch.float32,
                ).to(self.device.type),
                torch.randn(
                    (
                        self.batch_size,
                        self.num_frames,
                        (
                            self.in_channels
                            if self.residual_mode == ResidualMode.SUBTRACT_NEXT_FRAME
                            else 2
                        ),
                        224,
                        224,
                    ),
                    dtype=torch.float32,
                ).to(self.device.type),
            )

        self.learning_rate = learning_rate
        self.dl_classification_mode = dl_classification_mode
        self.eval_classification_mode = eval_classification_mode

        # TODO: Move this to setup() method.
        # Sets metric if None.
        self.dice_metrics = {}
        self.other_metrics = {}
        setup_metrics(self, metric, classes)

        # Attempts to load checkpoint if provided.
        self.weights_from_ckpt_path = weights_from_ckpt_path
        if self.weights_from_ckpt_path:
            ckpt = torch.load(self.weights_from_ckpt_path)
            try:
                self.load_state_dict(ckpt["state_dict"])
            except KeyError:
                # HACK: So that legacy checkpoints can be loaded.
                try:
                    new_state_dict = OrderedDict()
                    for k, v in ckpt.items():
                        name = k[7:]  # remove 'module.' of dataparallel
                        new_state_dict[name] = v
                    self.model.load_state_dict(  # pyright: ignore[reportAttributeAccessIssue]
                        new_state_dict
                    )
                except RuntimeError as e:
                    raise e

    def on_train_end(self) -> None:
        """Call at the end of training before logger experiment is closed."""
        if self.dump_memory_snapshot:
            torch.cuda.memory._dump_snapshot("attention_unet_snapshot.pickle")

    def forward(self, x_img: torch.Tensor, x_res: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # HACK: This is to get things to work with deepspeed opt level 1 & 2. Level 3
        # is broken due to the casting of batchnorm to non-fp32 types.
        with torch.autocast(device_type=self.device.type):
            return self.model(x_img, x_res)  # pyright: ignore[reportCallIssue]

    def on_train_epoch_end(self) -> None:
        """Call in the training loop at the very end of the epoch."""
        shared_metric_logging_epoch_end(self, "train")

    def on_validation_epoch_end(self) -> None:
        """Call in the validation loop at the very end of the epoch."""
        shared_metric_logging_epoch_end(self, "val")

    def on_test_epoch_end(self) -> None:
        """Call in the test loop at the very end of the epoch."""
        shared_metric_logging_epoch_end(self, "test")

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
    ):
        """Forward pass for the model with dataloader batches.

        Args:
            batch: Batch of frames, residual frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.

        Return:
            torch.tensor: Training loss.

        Raises:
            AssertionError: Prediction shape and ground truth mask shapes are different.

        """
        images, res_images, masks, _ = batch
        bs = images.shape[0] if len(images.shape) > 3 else 1
        images_input = images.to(self.device.type)
        res_input = res_images.to(self.device.type)
        masks = masks.to(self.device.type).long()

        with torch.autocast(device_type=self.device.type):
            masks_proba: torch.Tensor = self.model(
                images_input, res_input
            )  # pyright: ignore[reportCallIssue] # False positive

            if self.dl_classification_mode == ClassificationMode.MULTILABEL_MODE:
                # GUARD: Check that the sizes match.
                assert (
                    masks_proba.size() == masks.size()
                ), f"Output of shape {masks_proba.shape} != target shape: {masks.shape}"

            # HACK: This ensures that the dimensions to the loss function are correct.
            if isinstance(self.loss, nn.CrossEntropyLoss) or isinstance(
                self.loss, FocalLoss
            ):
                loss_seg = self.alpha * self.loss(masks_proba, masks.squeeze(dim=1))
            else:
                loss_seg = self.alpha * self.loss(masks_proba, masks)
            loss_all = loss_seg

        self.log(
            "loss/train",
            loss_all.item(),
            batch_size=bs,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"loss/train/{self.loss.__class__.__name__.lower()}",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
        )

        if isinstance(
            self.dice_metrics["train"], GeneralizedDiceScoreVariant
        ) or isinstance(self.dice_metrics["train"], MetricCollection):
            masks_preds, masks_one_hot = shared_metric_calculation(
                self, masks, masks_proba, "train"
            )

            if isinstance(self.logger, TensorBoardLogger):
                self._shared_image_logging(
                    batch_idx,
                    images.detach().cpu(),
                    masks_one_hot.detach().cpu(),
                    masks_preds.detach().cpu(),
                    "train",
                    10,
                )
            self.train()

        return loss_all

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
    ):
        """Forward pass for the model for one minibatch of a validation epoch.

        Args:
            batch: Batch of frames, residual frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.

        """
        self._shared_eval(batch, batch_idx, "val")

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
    ):
        """Forward pass for the model for one minibatch of a test epoch.

        Args:
            batch: Batch of frames, residual frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.

        """
        self._shared_eval(batch, batch_idx, "test")

    @torch.no_grad()
    def _shared_eval(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        prefix: Literal["val", "test"],
    ):
        """Shared evaluation step for validation and test steps.

        Args:
            batch: The batch of images and masks.
            batch_idx: The batch index.
            prefix: The runtime mode (val, test).

        """
        self.eval()
        images, res_images, masks, _ = batch
        bs = images.shape[0] if len(images.shape) > 3 else 1
        images_input = images.to(self.device.type)
        res_input = res_images.to(self.device.type)
        masks = masks.to(self.device.type).long()

        masks_proba: torch.Tensor = self.model(
            images_input, res_input
        )  # pyright: ignore[reportCallIssue] # False positive

        if self.dl_classification_mode == ClassificationMode.MULTILABEL_MODE:
            # GUARD: Check that the sizes match.
            assert (
                masks_proba.size() == masks.size()
            ), f"Output of shape {masks_proba.shape} != target shape: {masks.shape}"

        # HACK: This ensures that the dimensions to the loss function are correct.
        if isinstance(self.loss, nn.CrossEntropyLoss) or isinstance(
            self.loss, FocalLoss
        ):
            loss_seg = self.alpha * self.loss(masks_proba, masks.squeeze(dim=1))
        else:
            loss_seg = self.alpha * self.loss(masks_proba, masks)

        loss_all = loss_seg
        self.log(
            f"loss/{prefix}",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"loss/{prefix}/{self.loss.__class__.__name__.lower()}",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"hp/{prefix}_loss",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
        )

        if isinstance(
            self.dice_metrics[prefix], GeneralizedDiceScoreVariant
        ) or isinstance(self.dice_metrics[prefix], MetricCollection):
            masks_preds, masks_one_hot = shared_metric_calculation(
                self, masks, masks_proba, prefix
            )

            if isinstance(self.logger, TensorBoardLogger):
                self._shared_image_logging(
                    batch_idx,
                    images.detach().cpu(),
                    masks_one_hot.detach().cpu(),
                    masks_preds.detach().cpu(),
                    prefix,
                    10,
                )

    @torch.no_grad()
    def _shared_image_logging(
        self,
        batch_idx: int,
        images: torch.Tensor,
        masks_one_hot: torch.Tensor,
        masks_preds: torch.Tensor,
        prefix: Literal["train", "val", "test"],
        every_interval: int = 10,
    ):
        """Log the images to tensorboard.

        Args:
            batch_idx: The batch index.
            images: The input images.
            masks_one_hot: The ground truth masks.
            masks_preds: The predicted masks.
            prefix: The runtime mode (train, val, test).
            every_interval: The interval to log images.

        Return:
            None

        Raises:
            AssertionError: If the logger is not detected or is not an instance of
            TensorboardLogger.
            ValueError: If any of `images`, `masks`, or `masks_preds` are malformed.

        """
        assert self.logger is not None, "No logger detected!"
        assert isinstance(
            self.logger, TensorBoardLogger
        ), f"Logger is not an instance of TensorboardLogger, but is of type {type(self.logger)}"

        if batch_idx % every_interval == 0:
            # This adds images to the tensorboard.
            tensorboard_logger: SummaryWriter = self.logger.experiment
            match prefix:
                case "val" | "test":
                    step = int(
                        sum(self.trainer.num_val_batches) * self.trainer.current_epoch
                        + batch_idx
                    )
                case _:
                    step = self.global_step

            # NOTE: This will adapt based on the color mode of the images
            if self.loading_mode == LoadingMode.RGB:
                inv_norm_img = self.de_transform(images).detach().cpu()
            else:
                image = (
                    images[:, :, 0, :, :]
                    .unsqueeze(2)
                    .repeat(1, 1, 3, 1, 1)
                    .detach()
                    .cpu()
                )
                inv_norm_img = self.de_transform(image).detach().cpu()

            pred_images_with_masks = [
                draw_segmentation_masks(
                    img,
                    masks=mask.bool(),
                    alpha=0.7,
                    colors=["black", "red", "blue", "green"],
                )
                # Get only the first frame of images.
                for img, mask in zip(
                    inv_norm_img[:, 0, :, :, :].detach().cpu(),
                    masks_preds.detach().cpu(),
                    strict=True,
                )
            ]
            gt_images_with_masks = [
                draw_segmentation_masks(
                    img,
                    masks=mask.bool(),
                    alpha=0.7,
                    colors=["black", "red", "blue", "green"],
                )
                # Get only the first frame of images.
                for img, mask in zip(
                    inv_norm_img[:, 0, :, :, :].detach().cpu(),
                    masks_one_hot.detach().cpu(),
                    strict=True,
                )
            ]
            combined_images_with_masks = gt_images_with_masks + pred_images_with_masks

            tensorboard_logger.add_images(
                tag=f"{prefix}/preds",
                img_tensor=torch.stack(tensors=combined_images_with_masks, dim=0)
                .detach()
                .cpu(),
                global_step=step,
            )

    @torch.no_grad()
    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str | list[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Forward pass for the model for one minibatch of a test epoch.

        Args:
            batch: Batch of frames, residual frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.
            dataloader_idx: Index of the dataloader.

        Return:
            tuple[torch.tensor, torch.tensor, str]: Mask predictions, original images,
                and filename.

        """
        self.eval()
        images, res_images, masks, fn = batch
        images_input = images.to(self.device.type)
        res_input = res_images.to(self.device.type)
        masks = masks.to(self.device.type).long()

        masks_proba: torch.Tensor = self.model(
            images_input, res_input
        )  # pyright: ignore[reportCallIssue]

        if self.eval_classification_mode == ClassificationMode.MULTICLASS_MODE:
            masks_preds = masks_proba.argmax(dim=1)
            masks_preds = F.one_hot(masks_preds, num_classes=4).permute(0, -1, 1, 2)
        else:
            masks_preds = masks_proba > 0.5

        return masks_preds.detach().cpu(), images.detach().cpu(), fn

    @override
    def configure_optimizers(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return utils.configure_optimizers(self)
