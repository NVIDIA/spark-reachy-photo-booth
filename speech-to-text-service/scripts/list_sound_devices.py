# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sounddevice as sd


def list_devices(device_list, device_type: str = "input"):
    devices = [
        (i, device)
        for i, device in enumerate(device_list)
        if device.get(f"max_{device_type}_channels", 0) > 0
    ]

    if not devices:
        print(f"No {device_type} devices found.")
        return

    print(f"Available {device_type} devices:")
    print("-" * 40)
    for i, device in devices:
        channels = device.get(f"max_{device_type}_channels", 0)
        samplerate = device.get("default_samplerate", "Unknown")
        samplerate = f"{int(samplerate)} Hz" if isinstance(samplerate, int | float) else "Unknown"  # noqa: E501 # fmt: skip
        print(f"Index: {i:2d} | Name: {"'"+device.get('name', 'Unknown')+"'":<50} | Channel Count: {channels:2d} | Sample Rate: {samplerate}")  # noqa: E501 # fmt: skip


if __name__ == "__main__":
    try:
        devices = sd.query_devices()
    except sd.PortAudioError as e:
        print(f"Error querying audio devices: {e}")
        exit(1)

    list_devices(devices, "input")
    print("\n")
    list_devices(devices, "output")
