from __future__ import annotations

from json import decoder, encoder
from typing import Any, Callable, Literal, Sequence, override

import torch
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base.encoders import get_encoder
from segmentation_models_pytorch.base.heads import ClassificationHead, SegmentationHead
from segmentation_models_pytorch.base.model import SegmentationModel
from torch import nn
from torch.nn import functional as F


class OneD(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_frames: int,
    ) -> None:
        super().__init__()
        match num_frames:
            case 5:
                self.one = nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_channels, kernel_size=5, stride=5, padding=0
                    ),
                    nn.ReLU(),
                )
            case 10:
                self.one = nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_channels, kernel_size=5, stride=5, padding=0
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        out_channels, out_channels, kernel_size=2, stride=2, padding=0
                    ),
                    nn.ReLU(),
                )
            case 15:
                self.one = nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_channels, kernel_size=5, stride=5, padding=0
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        out_channels, out_channels, kernel_size=3, stride=3, padding=0
                    ),
                    nn.ReLU(),
                )
            case 20:
                self.one = nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_channels, kernel_size=5, stride=5, padding=0
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        out_channels, out_channels, kernel_size=4, stride=4, padding=0
                    ),
                    nn.ReLU(),
                )
            case 30:
                self.one = nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_channels, kernel_size=5, stride=5, padding=0
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        out_channels, out_channels, kernel_size=3, stride=3, padding=0
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        out_channels, out_channels, kernel_size=2, stride=2, padding=0
                    ),
                )
            case _:
                raise NotImplementedError(
                    f"Model with num_frames of {num_frames} not implemented!"
                )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.one(x)


def compress_2(stacked_outputs: torch.Tensor, block: OneD) -> torch.Tensor:
    """Apply the OneD temporal convolution on the stacked outputs.

    Args:
        stacked_outputs: 5D tensor of shape (num_frames, batch_size, num_channels, n, n).
        block: 1d temporal convolutional block.

    Return:
        torch.Tensor: 4D tensor of shape (batch_size, num_channels, n, n).
    """
    # -- (1) Swap axes. Output shape: (batch size, n * n, num_frames, num_channels) --
    reshaped_output0 = stacked_outputs.permute(1, 3, 4, 0, 2).flatten(1, 2)

    # -- (2) Reorder channels so that the first num_frames channels are the first
    # channel from the num_frames. --
    batch_size, n_n, _, _ = reshaped_output0.shape
    reordered_output = (
        reshaped_output0.permute(0, 1, 3, 2).contiguous().view(batch_size, n_n, -1)
    )

    # -- (3) Flatten with an output shape of (batch_size, 1, n * n * num_channels *
    # num_frames) --
    flattened_output = reordered_output.flatten(1, 2).unsqueeze(1)

    # -- (4) Apply the 1d temporal conv block. Output shape is (batch_size, n * n,
    # n * n * num_channels) --
    compressed_image = block(flattened_output)

    # -- (5) Average layers --
    channel_dim = 1
    averaged_img = compressed_image.mean(channel_dim).squeeze(channel_dim)

    # -- (6) Reshape the input shape of (batch_size, 1, n * n * num_channels) to
    # (batch_size, num_channels, n, n) --
    n = stacked_outputs.shape[-1]
    num_channels = stacked_outputs.shape[-3]

    # -- (6.1) Reshape (batch_size, 1, n * n * num_channels) to (batch_size,
    # num_channels, n, n) using unflatten. --
    final_output = averaged_img.unflatten(-1, (n * n, num_channels))

    # -- (6.2) Reshape (batch_size, n * n, num_channels) to (batch_size, n, n,
    # num_channels) using permute. --
    final_output = final_output.permute(0, 3, 1, 2)

    return final_output


class CustomSegmentationModel(SegmentationModel):
    @override
    def __init__(self, *args, num_frames: Literal[5, 10, 15, 20, 30], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_frames = num_frames

    @override
    def initialize(self):
        # Define encoders and segmentation head.
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            initialize_head(self.classification_head)

        # Create the 1D temporal conv blocks, with more output channels the more pixels
        # the output has.
        # At the first 1D layer, there will only be 49 pixels (7 x 7), so fewer output
        # channels are needed.
        # Ideally, every step should require x4 channels. Due to memory limitations,
        # this was not possible.

        # INFO: Parameter tuning for output channels.
        self.oned1 = OneD(1, 2, self.num_frames)
        self.oned2 = OneD(1, 5, self.num_frames)
        self.oned3 = OneD(1, 10, self.num_frames)
        self.oned4 = OneD(1, 20, self.num_frames)
        self.oned5 = OneD(1, 40, self.num_frames)

        self.onedlayers = [self.oned1, self.oned2, self.oned3, self.oned4, self.oned5]

    def check_input_shape(self, x: torch.Tensor) -> None:
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h & output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compressed_features = []

        # The first layer of the skip connection gets ignored, but in order for the
        # indexing later on to work, the feature output needs an empty first output.
        compressed_features.append(["EMPTY"])

        # Goes through each frame of the image and add the output features to a list.
        features_list = []
        if len(x) > 1:
            for img in x:
                self.check_input_shape(img)
                features = self.encoder(img)
                features_list.append(features)

            # Goes through each layer and gets the output from that layer from all the
            # feature outputs.
            for index in range(1, 6):
                layer_output = []
                for outputs in features_list:
                    image_output = outputs[index]
                    layer_output.append(image_output)
                layer_output = torch.stack(layer_output)

                # Define the 1D block with the correct number of output channels.
                block = self.onedlayers[index - 1]

                # Applies the compress_2 function to resize and reorder channels.
                compressed_output = compress_2(layer_output, block)
                compressed_features.append(compressed_output)

            # Send the compressed features up the decoder
            decoder_output = self.decoder(*compressed_features)

            # Apply segmentation head and return the prediction
            masks = self.segmentation_head(decoder_output)
            return masks

    @override
    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with
        `torch.no_grad()`.

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


class Unet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: list[int] = [256, 128, 64, 32, 16],
        decoder_attention_type: Literal["scse"] | None = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: str | Callable | None = None,
        aux_params: dict[str, Any] | None = None,
    ) -> None:
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
            centre=encoder_name.startswith("vgg"),
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
        self.initialize()


def initialize_decoder(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
