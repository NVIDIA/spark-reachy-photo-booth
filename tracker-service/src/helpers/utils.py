# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import torch


def from_cv2_to_torch(frame: np.ndarray) -> torch.Tensor:
    """Convert a frame from OpenCV format to PyTorch format.

    Args:
        frame (np.ndarray): The frame to convert. Shape: (H, W, 3) in BGR format
        in [0, 255].

    Returns:
        torch.Tensor: The converted frame. Shape: (1, 3, H, W).

    """
    H, W, C = frame.shape  # noqa: N806
    assert H > 0 and W > 0 and C == 3
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor_frame = torch.from_numpy(rgb_frame).unsqueeze(0).permute(0, 3, 1, 2)
    return tensor_frame


def from_torch_to_cv2(frame: torch.Tensor) -> np.ndarray:
    """Convert a frame from PyTorch format to OpenCV format.

    Args:
        frame (torch.Tensor): The frame to convert. Shape: (1, 3, H, W) in RGB format
        in [0, 1].

    Returns:
        np.ndarray: The converted frame. Shape: (H, W, 3) in BGR format in [0, 255].

    """
    _, C, H, W = frame.shape  # noqa: N806
    assert H > 0 and W > 0 and C == 3
    rgb_array = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
    bgr_frame = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_frame
