# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass

import numpy as np

type Degrees = float


@dataclass
class EulerAngle:
    roll: Degrees = 0.0
    pitch: Degrees = 0.0
    yaw: Degrees = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.roll, self.pitch, self.yaw])


@dataclass
class RobotPosition:
    head_rotation: EulerAngle = dataclasses.field(default_factory=EulerAngle)
    body_angle: Degrees = 0.0


@dataclass
class TrackingData:
    action_uuid: str
    slow_mode_distance_threshold: float
    fast_mode_distance_threshold: float
    slow_speed: float
    fast_speed: float
    robot_position: RobotPosition

    user_centered: bool = False
    paused: bool = False

    def __post_init__(self):
        if not (
            0.0
            <= self.slow_mode_distance_threshold
            < self.fast_mode_distance_threshold
            <= 1.0
        ):
            raise ValueError(
                f"Invalid thresholds: slow={self.slow_mode_distance_threshold}, "
                f"fast={self.fast_mode_distance_threshold}. "
                f"Must satisfy: 0.0 <= slow < fast <= 1.0"
            )

        if not (0.0 <= self.slow_speed < self.fast_speed):
            raise ValueError(
                f"Invalid speeds: slow={self.slow_speed}, fast={self.fast_speed}. "
                f"Must satisfy: 0.0 <= slow < fast"
            )


def convert_bits_per_sample(audio: np.ndarray, bits_per_sample: int) -> np.ndarray:
    if not isinstance(audio, np.ndarray):
        raise TypeError("Audio must be a numpy ndarray")

    if bits_per_sample == 8:
        return (audio * 127).astype(np.int8)
    elif bits_per_sample == 16:
        return (audio * 32767).astype(np.int16)
    elif bits_per_sample == 24:
        return (audio * 8388607).astype(np.int32)
    elif bits_per_sample == 32:
        return (audio * 2147483647).astype(np.int32)
    else:
        raise ValueError(f"Unsupported bit depth: {bits_per_sample}")
