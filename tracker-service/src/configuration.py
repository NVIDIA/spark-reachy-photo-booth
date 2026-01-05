# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field
from workmesh.config import BaseConfig


class Robot(Enum):
    """Robot name for the tracker service."""

    RESEARCHER = "researcher"


class FilterConfig(BaseModel):
    """Configuration for the filter."""

    name: str | None = Field(default=None, description="The name of the filter to use")
    alpha: float = Field(
        default=0.3, description="The alpha for the filter", ge=0.0, le=1.0
    )
    cutoff_frequency: float = Field(
        default=0.1, description="The cutoff frequency for the filter"
    )
    sample_rate: float = Field(
        default=2.0, description="The sample rate for the filter", ge=0.0
    )
    order: int = Field(default=1, description="The order for the filter", ge=1)
    min_alpha: float = Field(
        default=0.1, description="The minimum alpha for the filter", ge=0.0, le=1.0
    )
    max_alpha: float = Field(
        default=0.7, description="The maximum alpha for the filter", ge=0.0, le=1.0
    )
    sensitivity: float = Field(
        default=1.0, description="The sensitivity for the filter", ge=0.0
    )
    state_dim: int = Field(
        default=4, description="The state dimension for the filter", ge=1
    )
    measurement_dim: int = Field(
        default=2, description="The measurement dimension for the filter", ge=1
    )


class TrackerConfig(BaseConfig):
    """Configuration for the tracker service."""

    model_type: str = Field(
        default="models.detector.byte_track.ByteTrackDetector",
        description="The class of model to use. "
        "Note: this class must be a fully qualified path to the class",
    )
    robot_id: Robot = Field(
        default=Robot.RESEARCHER,
        description="The ID of the robot",
    )
    width: int = Field(
        default=512,
        description="The width of the image processed by the model",
        ge=1,
        le=1920,
    )
    height: int = Field(
        default=512,
        description="The height of the image processed by the model",
        ge=1,
        le=1080,
    )
    model: dict[str, Any] = Field(
        default={"confidence_threshold": 0.95, "iou_threshold": 0.2},
        description="The configuration for the model. "
        "Note: this configuration must be a dictionary of the model's configuration",
    )
    filter: FilterConfig = Field(
        default=FilterConfig(), description="The filter configuration to use"
    )
    sort_key: Literal["area", "score"] = Field(
        default="area",
        description=(
            "The key to sort the bounding box candidates by. "
            "'area' prefers largest person."
        ),
    )
    vicinity_threshold: float = Field(
        default=0.10,
        description="The sort key value the bounding box must be above to be "
        "considered to be in close proximity.",
        ge=0.0,
        le=1.0,
    )
    distance_threshold: float = Field(
        default=0.01,
        description="The sort key value the bounding box must be below to be "
        "considered to be far away.",
        ge=0.0,
        le=1.0,
    )
    marker_switch_threshold: float = Field(
        default=0.1,
        description="The minimum difference in sort key between the tracked detection "
        "and a new detection required to switch markers. For area: absolute difference "
        "(e.g., 0.05 = 5% of image). For score: confidence difference.",
        ge=0.0,
        le=1.0,
    )
    marker_switch_patience: float = Field(
        default=1000.0,
        description=(
            "The time in milliseconds a new detection must be consistently "
            "better than the tracked detection before switching markers. "
        ),
        ge=0.0,
    )
    fallback_method: Literal["last_frame", "center_frame"] = Field(
        default="last_frame",
        description="The fallback method to use. If no bounding box is detected, "
        "the fallback method will be used",
    )
    filtered_fields: list[str] = Field(
        default=["width", "height", "top_left_x", "top_left_y", "nose_x", "nose_y"],
        description="The fields to filter the bounding box by",
    )
    patience: float = Field(
        default=2000.0,
        description="The time in milliseconds after which no detection is found "
        "to be considered as absent.",
        ge=0.0,
    )
    fps: float = Field(
        default=30.0,
        description="The fps of the tracker service.",
        ge=1.0,
    )
    center_penalty_enabled: bool = Field(
        default=True,
        description=(
            "Whether to enable center penalty. Boxes at the edge of "
            "the frame will be penalized when selecting the best detection."
        ),
    )
    center_penalty_min: float = Field(
        default=0.6,
        description=(
            "Minimum penalty factor applied to detections at the edge "
            "of the frame. 1.0 = no penalty at center, this value = "
            "penalty at edge. E.g., 0.8 means boxes at the edge are "
            "weighted 80% of their actual area/score."
        ),
        ge=0.0,
        le=1.0,
    )
    center_penalty_type: Literal["linear", "quadratic", "cubic"] = Field(
        default="linear",
        description="The type of penalty function to use. "
        "'linear' = smooth linear falloff, "
        "'quadratic' = faster falloff (parabolic), "
        "'cubic' = even faster falloff.",
    )
