from itertools import product
from typing import Literal
import torch
import pytest

from models.two_plus_one import (
    OneD,
    RESNET_OUTPUT_SHAPES,
    compress_dilated,
    DilatedOneD,
)
from models.two_plus_one import compress_2 as _compress2_new

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class TestNewCompress:
    batch_size = 2
    num_channels = 128
    height = 112
    width = 112

    def _test_with_num_frames(self, num_frames: int):
        input_original = torch.randn(
            self.batch_size, num_frames, self.num_channels, self.height, self.width
        ).to(DEVICE)
        oned = OneD(1, 2, num_frames).to(DEVICE)
        oned.eval()

        try:
            old_compress_out = _compress_wrapper(input_original, oned)
            new_compress_out = _compress2_new(input_original, oned)
        except Exception as e:
            raise ExceptionGroup(f"Input of shape {input_original.shape}", [e]) from e

        try:
            allclose = torch.allclose(old_compress_out, new_compress_out)
        except RuntimeError as e:
            raise ExceptionGroup(
                f"Old shape: {old_compress_out.shape}, New shape: {new_compress_out.shape}",
                [e],
            ) from e

        assert allclose, "Values of the operations are not the same."

    def test_conv1d(self):
        """Test the Conv1D model with different approaches to compressing the input.

        This test is performed for frames in range 5 to 30, with a step of 5.
        """
        for num_frames in range(5, 35, 5):
            self._test_with_num_frames(num_frames)


class TestNewOneD:
    batch_size = 2
    resnets = ["resnet18", "resnet34", "resnet50"]
    num_frames = range(5, 35, 5)

    @pytest.mark.parametrize(
        "num_frames,resnet,layer", product(num_frames, resnets, range(5))
    )
    def test_dilated_conv1d(
        self,
        num_frames: int,
        resnet: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        layer: int,
    ):
        num_channels, height, width = RESNET_OUTPUT_SHAPES[resnet][layer]
        input_original = torch.randn(
            self.batch_size,
            num_frames,
            num_channels,
            height,
            width,
        ).to(DEVICE)

        torch.manual_seed(0)
        current_oned = OneD(1, 2, num_frames, False, "relu").to(DEVICE)
        torch.manual_seed(0)
        new_oned = DilatedOneD(1, 2, num_frames, height * width, False, "relu").to(
            DEVICE
        )

        try:
            current_out = _compress2_new(input_original, current_oned)
            new_out = compress_dilated(input_original, new_oned)
        except Exception as e:
            raise ExceptionGroup(f"Input of shape {input_original.shape}", [e]) from e

        try:
            allclose = torch.allclose(current_out, new_out)
        except RuntimeError as e:
            raise ExceptionGroup(
                f"Current shape: {current_out.shape}, New shape: {new_out.shape}", [e]
            ) from e

        assert allclose, (
            "Values of the operations are not the same; "
            + f"avg abs diff: {(new_out - current_out).abs().mean()}"
        )


def compress_2(stacked_outputs: torch.Tensor, block: OneD) -> torch.Tensor:
    """Compresses the input tensor using the one dimensional block.

    Args:
        stacked_outputs: Input tensor of shape (n_frames, batch_size, channels, h, w).
        block: One dimensional block.

    Return:
        Compressed output tensor.
    """
    # Input shape is (n_frames, batch_size, channels, n, n). Ensure that block has an
    # in_channels of one and and out_channels of n*n

    # Step 1: Swap axes. Output shape: (batch_size, n*n, n_frames, channels)
    reshaped_output0 = stacked_outputs.permute(1, 3, 4, 0, 2).flatten(1, 2)

    # Step 2: Reorder channels so that the first 30 channels, or however many frames,
    # are the first channel from the 30 frames. Output shape: (batch_size, n*n,
    # n_frames * channels)
    batch_size = reshaped_output0.shape[0]
    n_n = reshaped_output0.shape[1]
    reordered_output = (
        reshaped_output0.permute(0, 1, 3, 2).contiguous().view(batch_size, n_n, -1)
    )

    # Step 3: Flatten with an output shape of (batch_size, 1, n*n * channels * n_frames)
    flattened_output = reordered_output.flatten(1, 2).unsqueeze(1)

    # Step 4: Apply one dimensional block. Output shape is (batch_size, n*n ,n*n *
    # channels)
    compressed_image = block(flattened_output)

    # Step 5: Average layers
    channel_dim = 1
    averaged_img = compressed_image.mean(channel_dim).squeeze(channel_dim)

    # Step 6: Reshape the input shape of (batch_size, 1, n*n * channels) to (batch_size,
    # channels, n, n)

    n = stacked_outputs.shape[-1]
    channels = stacked_outputs.shape[-3]

    # 6.1: Reshape (batch_size, 1, n*n * channels) to (batch_size, n*n, channels) using
    # unflatten
    final_output = averaged_img.unflatten(-1, (n * n, channels))

    # 6.2: Reshape (batch_size, n*n, channels) to (batch_size, n, n, channels) using
    # unflatten
    final_output = final_output.unflatten(1, (n, n))

    # 6.3: Reshape (batch_size, n, n, channels) to (batch_size, channels, n, n) using
    # permute
    final_output = final_output.permute(0, 3, 1, 2)

    # Return outputs from each step
    return final_output


def _compress_wrapper(stacked_outputs: torch.Tensor, block: OneD) -> torch.Tensor:
    """Wrapper function for compress2.

    Args:
        stacked_outputs: Input tensor of shape (B, F, C, H, W).
        block: One dimensional block.

    Return
        Compressed output tensor.
    """
    # Incoming tensor of shape (B, F, C, H, W)
    # Input to compress2 must be of shape (F, B, C, H, W)
    reshaped_outputs = stacked_outputs.permute(1, 0, 2, 3, 4)
    return compress_2(reshaped_outputs, block)
