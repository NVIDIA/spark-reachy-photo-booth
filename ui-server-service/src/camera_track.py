# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import deque

import cv2
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame
from workmesh.messages import Frame


class StreamTrack(VideoStreamTrack):
    """
    A video track that streams camera frames from the workmesh camera service.
    """

    def __init__(self, width: int, height: int):
        """
        Initialize the stream track.

        Args:
            width: The width of the video.
            height: The height of the video.
            robot_id: The robot ID associated with the stream track.
        """

        super().__init__()
        self._queue = deque(maxlen=2)  # Smaller queue for lower latency
        self._width = width
        self._height = height

    def update_frame(self, frame: Frame):
        """
        Update the latest frame from camera service.

        Args:
            frame: The frame to update.
        """

        self._queue.append(frame)

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        if len(self._queue) == 0:
            frame_data = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        else:
            frame_bytes = np.frombuffer(self._queue[-1].data, dtype=np.uint8)
            frame_data = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
            frame_data = cv2.resize(frame_data, (self._width, self._height))

        frame = VideoFrame.from_ndarray(frame_data, format="bgr24")  # type: ignore
        frame.pts = pts
        frame.time_base = time_base

        return frame
