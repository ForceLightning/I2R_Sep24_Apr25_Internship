from __future__ import annotations

from collections import deque
from typing import Sequence

import cv2
import numpy as np
import PIL.ImageSequence as ImageSequence
import torch
from cv2 import typing as cvt
from matplotlib import animation
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms.v2 import functional as v2f

from utils.utils import InverseNormalize


def dense_optical_flow(video: Sequence[cvt.MatLike]) -> list[cvt.MatLike]:
    """Computes dense optical flow on the CPU (slow).

    Args:
        video: Video frames to calculate optical flow with. Must be greyscale.

    Return:
        list[cvt.MatLike]: Optical flow.

    Raises:
        AssertionError: Video is not greyscale.
    """
    video = deque(video)
    assert all(
        frame.ndim == 2 for frame in video
    ), f"Video with frame of input shape {video[0].shape} must have only 2 dims: (height, width)."
    frame_1 = video[0]
    frame_1 = frame_1.reshape(*frame_1.shape, 1)  # View as (H, W, 1)

    flows: list[cvt.MatLike] = []
    video.rotate(-1)
    for frame in video:
        next = frame.reshape(*frame.shape, 1)

        # NOTE: The type hints in CV2 seem not to allow NoneType values, while
        # the documentation clearly shows that they are supported. Thus, static
        # type checking may throw errors below.
        flow = cv2.calcOpticalFlowFarneback(  # pyright: ignore[reportCallIssue]
            frame_1,
            next,
            None,  # pyright: ignore[reportArgumentType] False positive
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        flows.append(flow)
        frame_1 = next

    return flows


def cuda_optical_flow(video: Sequence[cvt.MatLike]) -> list[cvt.MatLike]:
    """Computes dense optical flow with hardware acceleration on NVIDIA cards.

    This method computes optical flow between all frames i with i + 1 for up to
    sequence length n-1, then computes for frame n and frame 0.

    Args:
        video: Video frames to calculate optical flow with. Must be greyscale.

    Return:
        list[cvt.MatLike]: Optical flow.

    Raises:
        AssertionError: CUDA support is not enabled.
        AssertionError: Video is not greyscale.
        AssertionError: Video frames are not of the same shape.
        RuntimeError: Error in calculating optical flow (see stack trace).
    """
    num_cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    assert num_cuda_devices, (
        f"Number of enabled cuda devices found: {num_cuda_devices}, must be > 0"
        + "OpenCV-contrib-python must be built with CUDA support enabled."
    )

    seq_len = len(video)
    assert all(
        frame.ndim == 2 for frame in video
    ), f"Video with frame of input shape {video[0].shape} must have only 2 dims: (height, width)"
    cv2.cuda.resetDevice()
    h, w = video[0].shape

    assert all(
        frame.shape == video[0].shape and frame.dtype == np.uint8 for frame in video
    ), f"All frames must have shape: ({h}, {w}) and be of type `numpy.uint8`"

    # TODO: Add more options for optical flow calculation

    # Initialise NVIDIA Optical Flow (SDK 2)
    nvof = cv2.cuda.NvidiaOpticalFlow_2_0.create(
        (h, w),
        perfPreset=cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_SLOW,
        outputGridSize=cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_OUTPUT_VECTOR_GRID_SIZE_1,
        hintGridSize=cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_HINT_VECTOR_GRID_SIZE_1,
        enableTemporalHints=True,
    )

    # Allocate GPU memory for the GPU Matrices.
    cu_frame_1 = cv2.cuda.GpuMat()
    cu_frame_2 = cv2.cuda.GpuMat()

    res: list[cvt.MatLike] = []
    for i in range(seq_len):

        # Transfer frames to the GPU.
        cu_frame_1.upload(video[i])
        cu_frame_2.upload(video[(i + 1) % seq_len])

        try:
            # NOTE: The type hints in CV2 seem not to allow NoneType values, while
            # the documentation clearly shows that they are supported. Thus, static
            # type checking may throw errors below.

            # Calculate flow.
            flow, _ = nvof.calc(  # pyright: ignore[reportCallIssue]
                cu_frame_1, cu_frame_2, None  # pyright: ignore[reportArgumentType]
            )

            # Convert the value back to float, transfer memory to CPU and append to res
            res.append(
                nvof.convertToFloat(  # pyright: ignore[reportCallIssue]
                    flow, None  # pyright: ignore[reportArgumentType]
                )
                .download()
                .astype(np.float32)
            )

        except Exception as e:
            raise RuntimeError(f"Error at iteration {i}") from e

    # Ensure GPU memory is fully released
    cu_frame_1.release()
    cu_frame_2.release()
    nvof.collectGarbage()

    return res


# TODO: Remove this when done, for debugging purposes.
def _plot_animated_with_flow(
    imgs,
    flows,
    inv_norm: InverseNormalize = InverseNormalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    ),
    show_plot: bool = True,
):
    fig, ax = plt.subplots()

    ims = []
    if isinstance(imgs, Image.Image):
        for i, image in enumerate(ImageSequence.Iterator(imgs)):
            im = ax.imshow(image, animated=True)
            if i == 0:
                ax.imshow(image)
            ims.append([im])
    else:
        for i in range(len(imgs)):
            img = imgs[i]
            flow = flows[i]
            c, h, w = flow.shape
            if c == 2:
                flow = flows[i].permute(1, 2, 0).detach().cpu().numpy()
                hsv = np.zeros((h, w, 3), dtype=np.uint8)
                hsv[..., 1] = 255
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2

                # NOTE: The type hints in CV2 seem not to allow NoneType values, while
                # the documentation clearly shows that they are supported. Thus, static
                # type checking may throw errors below.
                hsv[..., 2] = cv2.normalize(  # pyright: ignore[reportCallIssue]
                    mag,
                    None,  # pyright: ignore[reportArgumentType]
                    0,
                    255,
                    cv2.NORM_MINMAX,
                )
                flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # if i == 14:
                #     sns.displot(x=hsv[...,0].reshape(-1), kde=True)
                #     plt.show()
            else:
                flow = np.asarray(v2f.to_pil_image(flow))

            img = np.asarray(v2f.to_pil_image(inv_norm(img).clamp(0, 1)))  # (H, W, C)
            combined_img = cv2.addWeighted(img, 1.0, flow, 0.85, 0)

            im = ax.imshow(combined_img, animated=True)
            if i == 0:
                ax.imshow(combined_img)
            ims.append([im])
    ani = animation.ArtistAnimation(
        fig, ims, interval=100 / 3, blit=True, repeat_delay=0
    )
    if show_plot:
        plt.show()
    return ani


# TODO: Remove this when done debugging.
if __name__ == "__main__":
    CINE_PATH = "data/train_val/Cine/4_1_0000.nii.tiff"

    _, cine_list = cv2.imreadmulti(CINE_PATH, flags=cv2.IMREAD_GRAYSCALE)
    flow_list = cuda_optical_flow(cine_list)

    combined_video = torch.empty((30, 512, 512), dtype=torch.uint8)
    for i in range(30):
        img = cine_list[i]
        img = cv2.resize(img, (512, 512))
        combined_video[i, :, :] = torch.as_tensor(img)
    combined_video = combined_video.view(30, 1, 512, 512)

    combined_flow = torch.empty((30, 512, 512, 2), dtype=torch.uint8)
    for i in range(30):
        img = flow_list[i]
        img = cv2.resize(img, (512, 512))
        combined_flow[i, :, :, :] = torch.as_tensor(img)
    combined_flow = combined_flow.permute(0, 3, 1, 2)

    _plot_animated_with_flow(combined_video, combined_flow)
