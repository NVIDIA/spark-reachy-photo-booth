# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Self

from clips.animations import Frame
from models import Range
from pydantic import BaseModel, Field, model_validator
from workmesh.config import BaseConfig
from workmesh.messages import Robot as RobotProto

SAMPLE_RATES = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000]
BITS_PER_SAMPLE = [8, 16, 24, 32]


class Robot(Enum):
    """Robot name for the animation compositor service."""

    RESEARCHER = "researcher"

    def to_proto(self) -> RobotProto:
        if self == Robot.RESEARCHER:
            return RobotProto.RESEARCHER
        raise ValueError(f"Unknown robot: {self}")


class AudioConfig(BaseModel):
    output_devices_index_or_name: list[int | str] = Field(
        default=["Reachy Mini Audio", "reSpeaker"]
    )

    sample_rate: int = Field(ge=1, lt=100000, default=16000)
    bits_per_sample: int = Field(ge=1, lt=100, default=16)
    channel_count: int = Field(ge=1, le=2, default=1)

    @model_validator(mode="after")
    def bits_per_sample_validator(self) -> Self:
        if self.bits_per_sample not in BITS_PER_SAMPLE:
            raise ValueError(
                f"Wrong bits_per_sample: {self.bits_per_sample}. "
                + f"Must be one of {BITS_PER_SAMPLE}"
            )
        return self

    @model_validator(mode="after")
    def sample_rate_validator(self) -> Self:
        if self.sample_rate not in SAMPLE_RATES:
            raise ValueError(
                f"Wrong sample_rate: {self.sample_rate}. Must be one of {SAMPLE_RATES}"
            )
        return self


class CompositorConfig(BaseConfig):
    robot_id: Robot = Field(
        default=Robot.RESEARCHER,
        description="The ID of the robot for this animation compositor. "
        "Note: this ID identifies the robot in the system",
    )
    frame_rate: int = Field(
        ge=1, lt=100, default=30, description="Frame rate for the animation compositor."
    )
    audio_config: AudioConfig = Field(
        default=AudioConfig(),
        description="Audio configuration for the animation compositor.",
    )
    offset_pose: Frame = Field(
        default=Frame.reference_pose(),
        description="Defines what is the offset pose of the robot.",
    )
    joint_limits: dict[str, Range] = Field(
        default={
            "body_angle": Range(min=-170, max=170),
            "r_antenna_angle": Range(min=-170, max=170),
            "l_antenna_angle": Range(min=-170, max=170),
            "head_rotation_roll": Range(min=-20, max=20),
            "head_rotation_pitch": Range(min=-30, max=30),
            "head_rotation_yaw": Range(min=-170, max=170),
        },
        description="Joint limits for the animation compositor.",
    )
    max_delta_per_frame: dict[str, float] = Field(
        default={
            "body_angle": 75.0,  # degrees per frame
            "r_antenna_angle": 20.0,
            "l_antenna_angle": 20.0,
            "head_position_x": 5.0,  # centimeters per frame
            "head_position_y": 5.0,
            "head_position_z": 5.0,
            "head_rotation": 75.0,  # degrees per frame for Euler angles
        },
        description="Maximum delta per frame for the animation compositor.",
    )
