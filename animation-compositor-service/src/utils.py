# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import sounddevice as sd
from clips.animations import (
    LINEAR_JOINT_NAMES,
    SPHERICAL_JOINT_NAMES,
    Animation,
    EulerAngles,
    Frame,
)
from configuration import AudioConfig, Robot
from workmesh.messages import (
    AnimationData,
    AnimationFrame,
    Audio,
    EulerAngle,
    RobotFrame,
)


def create_animation_frames(animation_data: AnimationData) -> Animation:
    """Create animation frames from message."""

    animation_frames: list[Frame] = []
    for frame in animation_data.frames:
        frame_kwargs = {}

        for joint in LINEAR_JOINT_NAMES:
            if frame.HasField(joint):
                frame_kwargs[joint] = getattr(frame, joint)

        for joint in SPHERICAL_JOINT_NAMES:
            if frame.HasField(joint):
                angles: EulerAngles = getattr(frame, joint)
                frame_kwargs[joint] = EulerAngles(
                    roll=angles.roll,
                    pitch=angles.pitch,
                    yaw=angles.yaw,
                )

        animation_frames.append(Frame(**frame_kwargs))

    return Animation(frames=animation_frames)


def create_robot_message(robot_id: Robot, frame: Frame, timestamp: float) -> RobotFrame:
    """Create robot message from frame."""

    head_rotation = (
        None
        if frame.head_rotation is None
        else EulerAngle(
            roll=frame.head_rotation.roll,
            pitch=frame.head_rotation.pitch,
            yaw=frame.head_rotation.yaw,
        )
    )

    return RobotFrame(
        robot_id=robot_id.to_proto(),
        timestamp=int(timestamp * 1000),
        frame=AnimationFrame(
            r_antenna_angle=frame.r_antenna_angle,
            l_antenna_angle=frame.l_antenna_angle,
            body_angle=frame.body_angle,
            head_position_x=frame.head_position_x,
            head_position_y=frame.head_position_y,
            head_position_z=frame.head_position_z,
            head_rotation=head_rotation,
        ),
    )


def validate_audio_settings(audio_msg: Audio, config: AudioConfig) -> bool:
    """Validate audio settings."""

    return (
        audio_msg.sample_rate == config.sample_rate
        and audio_msg.bits_per_sample == config.bits_per_sample
        and audio_msg.channel_count == config.channel_count
    )


def find_output_device_index(
    devices_id_or_name: list[int | str],
) -> tuple[bool, int, str]:
    """Find the index of the output device.

    Args:
        devices_id_or_name: The index or name of the possible output devices

    Returns:
        tuple[bool, int, str]: Whether one of the devices were found and
        the index and name of the device, or the default device index and
        name if none of the devices were found.
    """

    # Get all devices and filter for output devices
    devices = sd.query_devices()
    output_devices = []
    for d in devices:
        try:
            max_output_channels = (
                d["max_output_channels"]
                if isinstance(d, dict)
                else getattr(d, "max_output_channels", 0)
            )
        except sd.PortAudioError:
            max_output_channels = 0
        if isinstance(max_output_channels, int | float) and max_output_channels > 0:
            output_devices.append(d)
    devices = output_devices

    for device_id_or_name in devices_id_or_name:
        if isinstance(device_id_or_name, int):
            for device in devices:
                if device["index"] == device_id_or_name:
                    return True, device_id_or_name, device["name"]

        if isinstance(device_id_or_name, str):
            for device in devices:
                if device["name"] == device_id_or_name:
                    return True, device["index"], device["name"]

            # If no device is found, check if the device name contains the search string
            for device in devices:
                if device_id_or_name.lower() in device["name"].lower():
                    return True, device["index"], device["name"]

    # Return default output device if not found
    default_device = sd.query_devices(kind="output")
    if isinstance(default_device, dict):
        return (
            False,
            default_device.get("index", 0),
            default_device.get("name", "Unknown"),
        )
    else:
        # Fallback in case query_devices returns a list or unexpected type
        return False, 0, "Unknown"


def get_format_from_width(width: int, unsigned: bool = True) -> np.dtype:
    if width == 1:
        if unsigned:
            return np.dtype(np.uint8)
        return np.dtype(np.int8)
    if width == 2:
        return np.dtype(np.int16)
    if width == 3:
        return np.dtype(np.int32)
    if width == 4:
        return np.dtype(np.float32)

    raise ValueError(f"Invalid width: {width}")
