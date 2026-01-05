# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator
from reachy_mini.media.audio_control_utils import PARAMETERS as RESPEAKER_PARAMETERS
from reachy_mini.media.media_manager import MediaBackend
from workmesh.config import BaseConfig
from workmesh.messages import Robot as RobotProto


class Robot(Enum):
    """Robot name for the animation compositor service."""

    RESEARCHER = "researcher"

    def to_proto(self) -> RobotProto:
        if self == Robot.RESEARCHER:
            return RobotProto.RESEARCHER
        raise ValueError(f"Invalid robot: {self}")


class ReachyConfig(BaseModel):
    localhost_only: bool = Field(
        default=True, description="Whether to only connect to localhost daemons."
    )
    use_sim: bool = Field(default=False, description="Whether to use the simulation.")
    timeout: float = Field(default=5.0, description="The timeout for the connection.")
    daemon_wait_time: float = Field(
        default=3.0, description="The wait time for the daemon to initialize."
    )
    media_backend: MediaBackend = Field(default=MediaBackend.DEFAULT_NO_VIDEO, description="The media backend to use.")  # noqa: E501 # fmt: skip
    hardware_config_filepath: str | None = Field(
        default="/app/data/hardware_config.yaml",
        description="The filepath to the hardware configuration file.",
    )


class RespeakerConfig(BaseModel):
    # Microphone configuration
    microphone_parameters: dict[str, Any] = Field(
        default={"PP_ATTNS_MODE": [1]},
        description="The parameters for the ReSpeaker microphone.",
    )

    # DoA configuration
    enable_doa: bool = Field(default=False, description="Whether to enable the DoA.")
    doa_interval: float = Field(
        default=0.2, description="The interval in seconds to check the DoA."
    )

    @model_validator(mode="after")
    def validate_respeaker_config(self) -> Self:
        for key in self.microphone_parameters:
            if key not in RESPEAKER_PARAMETERS:
                raise ValueError(f"Invalid ReSpeaker parameter: {key}")
        return self


class RobotControllerConfig(BaseConfig):
    robot_id: Robot = Field(
        default=Robot.RESEARCHER,
        description="The ID of the robot for this robot controller.",
    )
    frame_rate_log_interval: float = Field(
        default=5.0, description="The interval in seconds to log the frame rate."
    )
    reachy_config: ReachyConfig = ReachyConfig()
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
    respeaker_config: RespeakerConfig = RespeakerConfig()
