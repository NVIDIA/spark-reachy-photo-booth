# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from dataclasses import dataclass, field

import sounddevice as sd
from workmesh.messages import UserUtteranceStatus


@dataclass
class Transcription:
    started: bool = False
    action_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    transcript: str = ""
    status: UserUtteranceStatus | None = None


def find_device_index(devices_id_or_name: list[int | str]) -> tuple[bool, int, str]:
    """Find the index of the device.

    Args:
        device_id_or_name: The index or name of the device

    Returns:
        tuple[bool, int, str]: Whether the device was found and the index and name
        of the device, or the default device index and name if the device was not found.
    """

    # Get all devices and filter for input devices
    devices = sd.query_devices()
    input_devices = []
    for device in devices:
        try:
            max_input_channels = (
                device["max_input_channels"]
                if isinstance(device, dict)
                else getattr(device, "max_input_channels", 0)
            )
        except Exception:
            max_input_channels = 0
        if isinstance(max_input_channels, int | float) and max_input_channels > 0:
            input_devices.append(device)
    devices = input_devices

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

    # Return default input device if not found
    default_device = sd.query_devices(kind="input")
    if isinstance(default_device, dict):
        return (
            False,
            default_device.get("index", 0),
            default_device.get("name", "Unknown"),
        )
    else:
        # Fallback in case query_devices returns a list or unexpected type
        return False, 0, "Unknown"
