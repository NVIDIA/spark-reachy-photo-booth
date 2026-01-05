# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from utils import Degrees, Position
from workmesh.config import BaseConfig


class Range(BaseModel):
    min: float = Field(ge=0, le=100)
    max: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_range(self) -> "Range":
        if self.min > self.max:
            raise ValueError(f"Invalid range: {self.min} > {self.max}")
        return self


class VoiceConfig(BaseModel):
    pitch_shift: float = Field(default=-0.1, description="Voice pitch shift factor")
    word_speed_range: float = Field(
        default=0.6, gt=0, description="Word speed variation range"
    )
    skip_chance: float = Field(
        default=0.1, ge=0, le=1, description="Probability of skipping words (0.0-1.0)"
    )


class TrackingConfig(BaseModel):
    """Configuration for the tracking"""

    slow_mode_distance_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="The distance from the center to enable slow tracking mode",
    )
    fast_mode_distance_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="The distance from the center to enable fast tracking mode",
    )
    slow_speed: float = Field(
        ge=0.0,
        default=2.0,
        description="The speed to move when the user is between slow and fast distance thresholds",  # noqa: E501
    )
    fast_speed: float = Field(
        ge=0.0,
        default=3.0,
        description="The speed to move when the user is beyond the fast distance threshold",  # noqa: E501
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> "TrackingConfig":
        if self.slow_mode_distance_threshold >= self.fast_mode_distance_threshold:
            raise ValueError(
                f"Slow mode distance threshold must be less than fast mode distance threshold: "  # noqa: E501
                f"{self.slow_mode_distance_threshold} >= {self.fast_mode_distance_threshold}"  # noqa: E501
            )
        if self.slow_speed >= self.fast_speed:
            raise ValueError(
                "Slow speed must be less than fast speed: "
                f"{self.slow_speed} >= {self.fast_speed}"
            )
        return self


class AnimationConfig(BaseModel):
    """Configuration for the animation"""

    # Find animation
    find_angle_step: Degrees = Field(
        default=45.0, description="The step angle to use when moving to find the user"
    )
    find_angle_pause: float = Field(
        default=0.5,
        description="The pause time between each angle step when finding the user in seconds",  # noqa: E501
    )
    find_range: Degrees = Field(
        ge=0.0,
        le=360.0,
        default=180.0,
        description="The range to search for the user in degrees. For example, "
        "180 degrees mean 90 degrees to the left and 90 degrees to the right",
    )

    # Tracking animation
    tracking: TrackingConfig = Field(
        default_factory=TrackingConfig,
        description="The configuration for the tracking animation",
    )


class LightConfig(BaseModel):
    """Configuration for the light animations"""

    focus_duration: float = Field(
        default=50.0,
        ge=0.0,
        description="Duration of fill circle animation in focus state (seconds)",
    )
    quick_transition_duration: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Duration for quick color transitions between animations (seconds)",
    )
    slow_transition_duration: float = Field(
        default=1.0,
        ge=0.0,
        le=3.0,
        description="Duration for slow color transitions between animations (seconds)",
    )
    picture_preparation_duration: float = Field(
        default=7.25,
        ge=0.0,
        le=30.0,
        description="Duration for picture preparation fade-in animation (seconds)",
    )
    take_picture_duration: float = Field(
        default=4.46,
        ge=0.0,
        le=30.0,
        description="Duration for take picture fade-out animation (seconds)",
    )


class RobotConfig(BaseModel):
    """Configuration for the robot"""

    voice_config: VoiceConfig = Field(default_factory=VoiceConfig)


class PhotoBoothBotConfig(RobotConfig):
    """Configuration for the photo bot"""

    animation_config: AnimationConfig = Field(default_factory=AnimationConfig)
    light_config: LightConfig = Field(default_factory=LightConfig)
    center_user_timeout: float = Field(
        default=10.0,
        ge=0.0,
        description="The timeout to center the user in seconds",
    )

    enable_listening_while_speaking: bool = Field(
        default=False,
        description="Whether to enable listening while the robot is speaking. \
        If using a speaker close to the mic, you risk capturing the robot's speech \
        as user speech, thus turning this on could mess up ASR. Turn it on if the \
        mic is not at risk of capturing the robot's speech. When turned on, only \
        utterances that finish after the robot stopped speaking will be processed. \
        Otherwise, they'll be ignored",
    )

    # Testing purposes
    enable_listen_animation: bool = Field(
        default=True, description="Whether to trigger the listen animation"
    )


class InteractionManagerConfig(BaseConfig):
    """Configuration for the interaction manager service."""

    robot_utterances_path: Path = Field(
        description="The path to the bot utterances file",
        default=Path("/app/data/robot_utterances.yaml"),
    )

    room_mapping: dict[str, Position] = Field(
        description="The mapping of the room objects",
        default={
            "screen": Position(x=-1.5, y=1.2),
        },
    )
    tool_names: dict[str, list[str]] = Field(
        description="The names of the tools",
        default={
            "start": ["greet_user"],
            "human": ["ask_human"],
            "image_generation": ["generate_image"],
            "taking_picture": ["look_at_human"],
            "end": ["farewell_user"],
        },
    )

    robot_config: dict[str, PhotoBoothBotConfig] = Field(
        default={"photo_booth_bot": PhotoBoothBotConfig()}
    )

    # NOTE: Doesn't take into account robot utterances, only normal clips
    global_clip_volume: float = Field(
        default=1, ge=0, le=1, description="The global volume of the clips"
    )

    time_between_comments: Range = Field(
        default=Range(min=1.5, max=3.0),
        description="The range of time between image generation comments in seconds. \
        Each comment break is a random value between the minimum and maximum.",
    )

    comment_look_direction: str = Field(
        default="center",
        description="The direction the robot should look when making comments. \
        Options: 'left', 'right', 'center'",
    )
