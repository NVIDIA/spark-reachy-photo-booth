# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from ipaddress import IPv4Address

from pydantic import BaseModel, Field, IPvAnyAddress
from workmesh.config import BaseConfig


class Robot(Enum):
    """Robot name for the camera service."""

    RESEARCHER = "researcher"


class ImageEncoding(Enum):
    """Image encoding for the camera service."""

    JPEG = ".jpg"
    PNG = ".png"
    BMP = ".bmp"
    RAW = ".raw"


class Resolution(BaseModel):
    """Resolution for the camera service."""

    width: int = Field(
        default=1920,
        description="The width of the image sent by the camera",
        ge=1,
        le=1920,
    )
    height: int = Field(
        default=1080,
        description="The height of the image sent by the camera",
        ge=1,
        le=1080,
    )


class CameraConfig(BaseConfig):
    """Configuration for the camera service."""

    camera_serial_or_url: str = Field(
        default="SN0001",
        description="The serial number or the url of the camera to use",
    )
    robot_id: Robot = Field(
        default=Robot.RESEARCHER,
        description="The ID of the robot to be mapped to the camera. "
        "Note: this ID identifies the robot in the system",
    )
    fps: int = Field(
        default=30,
        description="The number of frames per second to capture",
        ge=1,
        le=120,
    )
    streaming_resolution: Resolution = Field(
        default=Resolution(width=1920, height=1080),
        description="The resolution for streaming video from the camera",
    )
    capture_resolution: Resolution = Field(
        default=Resolution(width=1920, height=1080),
        description="The resolution for single shot photo from the camera",
    )
    encoding: ImageEncoding = Field(
        default=ImageEncoding.JPEG,
        description="The encoding of the image sent by the camera",
    )
    enable_http_server: bool = Field(
        default=True, description="Whether to enable the HTTP server"
    )
    http_bind_address: IPvAnyAddress = Field(
        default=IPv4Address("0.0.0.0"),
        description="The address to bind the HTTP server to",
    )
    http_bind_port: int = Field(
        default=7071, description="The port to bind the HTTP server to"
    )
    mirror_horizontal: bool = Field(
        default=False,
        description="Whether to mirror the image sent by the camera",
    )
