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
    video_deque = deque(video)
    assert (
        video_deque[0].ndim == 2
    ), f"video of input shape {video_deque[0].shape} must have 2 dims: (height, width)."
    frame_1 = video_deque[0]
    frame_1 = frame_1.reshape(*frame_1.shape, 1)
    prvs = (
        cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY) if frame_1.shape[2] == 3 else frame_1
    )

    hsv = np.zeros_like(frame_1).repeat(3, 2)
    hsv[..., 1] = 255
    flows: list[cvt.MatLike] = []
    video_deque.rotate(-1)
    for frame in video_deque:
        frame = frame.reshape(*frame.shape, 1)
        next = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame_1.shape[2] == 3 else frame
        )
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # NOTE: For visualisation purposes only
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        flows.append(bgr)
        frame_1 = next

    return flows


def cuda_optical_flow(video: Sequence[cvt.MatLike]):
    if not cv2.cuda.getCudaEnabledDeviceCount():
        raise RuntimeError(f"No CUDA-capable device is detected")

    video_deque = deque(video)
    assert (
        video[0].ndim == 2
    ), f"video frame of input shape {video[0].shape} must have 2 dims: (height, width)"

    frame_1 = video_deque[0]
    cu_frame_1 = cv2.cuda_GpuMat(frame_1)

    nvof = cv2.cuda_NvidiaOpticalFlow_1_0.create(
        (frame_1.shape[1], frame_1.shape[0]), 5, False, False, False, 0
    )

    video_deque.rotate(-1)
    res = []

    for frame in video_deque:
        cu_frame_2 = cv2.cuda_GpuMat(frame)
        flow, _ = nvof.calc(cu_frame_1, cu_frame_2, None)
        flow_upsampled = nvof.upSampler(
            flow, (frame_1.shape[1], frame_1.shape[0]), nvof.getGridSize(), None
        )
        nvof.collectGarbage()
        res.append(flow_upsampled)
        cu_frame_1 = cu_frame_2

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
            img = np.asarray(
                v2f.to_pil_image(inv_norm(imgs[i]).clamp(0, 1))
            )  # (H, W, C)
            flow = np.asarray(
                v2f.to_pil_image(inv_norm(flows[i]).clamp(0, 1))
            )  # (H, W, C)

            combined_img = cv2.addWeighted(img, 1.0, flow, 0.75, 0)

            im = ax.imshow(combined_img, animated=True)
            if i == 0:
                ax.imshow(np.asarray(v2f.to_pil_image(inv_norm(imgs[i]).clamp(0, 1))))
            ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=0)
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
