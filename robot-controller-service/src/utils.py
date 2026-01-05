# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import time
from logging import Logger
from typing import Any

import numpy as np
import psutil
from reachy_mini.media.audio_control_utils import PARAMETERS as RESPEAKER_PARAMETERS
from reachy_mini.media.audio_control_utils import init_respeaker_usb
from workmesh.messages import AnimationFrame, EulerAngle


def remap_head_translation(
    frame: AnimationFrame, scale=(1.0, 1.0, 1.0), translation_offset=(0.0, 0.0, 0.0)
):
    """
    Remaps Reachy animation frame values to configuration, adjusting for
    different translation measures in head position as needed.

    Args:
        frame: robot frame
        scale: (sx, sy, sz) tuple to scale head translation
        translation_offset: (ox, oy, oz) tuple to offset head translation

    Returns:
        tuple with (x, y, z) head position
    """

    return (
        frame.head_position_x * scale[0] + translation_offset[0],
        frame.head_position_y * scale[1] + translation_offset[1],
        frame.head_position_z * scale[2] + translation_offset[2],
    )


def clip_frame_by_max_delta(
    frame: AnimationFrame,
    current_position: AnimationFrame,
    max_delta_per_frame: dict[str, float],
) -> tuple[AnimationFrame, bool]:
    """
    Clips each joint value in the target AnimationFrame to ensure the change relative to
    the current position does not exceed the specified maximum delta per frame.
    This prevents sudden, unrealistic jumps by enforcing smooth transitions toward
    the target pose.

    Args:
        frame (AnimationFrame): The target AnimationFrame containing the desired joint
            values.
        current_position (AnimationFrame): The current AnimationFrame representing the
            robot's current joint values.
        max_delta_per_frame (dict[str, float]): A mapping from joint name to the
            maximum allowed change (delta) per frame for that joint.

    Returns:
        tuple[AnimationFrame, bool]:
            - A new AnimationFrame with each joint's value clipped if it exceeded
                the maximum delta.
            - A boolean indicating whether any clipping occurred (True if any value was
                clipped, False otherwise).
    """

    new_frame = AnimationFrame()
    was_clipped = False

    for joint, max_delta in max_delta_per_frame.items():
        # Handle float (linear) joints
        if joint != "head_rotation":
            target_value = getattr(frame, joint)
            current_value = getattr(current_position, joint)
            if isinstance(target_value, float) and isinstance(current_value, float):
                delta = target_value - current_value
                clipped_delta = np.clip(delta, -max_delta, max_delta)
                setattr(new_frame, joint, current_value + clipped_delta)

                if abs(delta) > max_delta:
                    was_clipped = True

        # Handle spherical head_rotation joint
        elif hasattr(frame, "head_rotation") and isinstance(
            frame.head_rotation, EulerAngle
        ):
            result_angles = {}
            for angle_name in ["roll", "pitch", "yaw"]:
                target_angle = getattr(frame.head_rotation, angle_name)
                current_angle = getattr(current_position.head_rotation, angle_name)
                if target_angle is not None and current_angle is not None:
                    delta = target_angle - current_angle

                    # Normalize to [-180, 180] (for degrees)
                    while delta > 180:
                        delta -= 360
                    while delta < -180:
                        delta += 360

                    if abs(delta) > max_delta:
                        was_clipped = True

                    clipped_delta = np.clip(delta, -max_delta, max_delta)
                    result_angles[angle_name] = current_angle + clipped_delta
                else:
                    result_angles[angle_name] = target_angle
            # Create a new EulerAngle instance properly
            new_frame.head_rotation.CopyFrom(EulerAngle(**result_angles))
        else:
            setattr(new_frame, joint, getattr(frame, joint))
    return new_frame, was_clipped


def frame_to_string(self: AnimationFrame) -> str:
    """Return a human-readable string representation of an AnimationFrame instance."""

    return (
        "{'body_angle': "
        + f"{self.body_angle}, "
        + f"'r_antenna_angle': {self.r_antenna_angle}, "
        + f"'l_antenna_angle': {self.l_antenna_angle}, "
        + f"'head_position_x': {self.head_position_x}, "
        + f"'head_position_y': {self.head_position_y}, "
        + f"'head_position_z': {self.head_position_z}, "
        + f"'head_rotation': {{'roll': {self.head_rotation.roll}, "
        + f"'pitch': {self.head_rotation.pitch}, "
        + f"'yaw': {self.head_rotation.yaw}}}"
        + "}}"
    )


AnimationFrame.__str__ = frame_to_string


# TODO: remove this when Pollen adds a way to use hardware config
# directly in the ReachyMini class
def start_daemon(
    logger: Logger,
    use_sim: bool,
    wait_time: float = 3.0,
    hardware_config_filepath: str | None = None,
) -> None:
    """Check if the Reachy Mini daemon is running and spawn it if necessary."""

    def is_python_script_running(
        script_name: str,
    ) -> tuple[bool, int | None, bool | None]:
        """
        Check if a specific Python script is running.

        Returns:
            tuple:
                - bool: True if the script is found running, else False
                - int | None: The PID of the process if found, else None
                - bool | None: True if the "--sim" flag is present (simulation enabled),
                               False if not, or None if undetermined
        """
        found_script = False
        simulation_enabled = False
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                for cmd in proc.info["cmdline"]:
                    if script_name in cmd:
                        found_script = True
                    if "--sim" in cmd:
                        simulation_enabled = True
                if found_script:
                    return True, proc.pid, simulation_enabled
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return False, None, None

    daemon_is_running, pid, sim = is_python_script_running("reachy-mini-daemon")
    if daemon_is_running and sim == use_sim:
        logger.info(
            f"[DAEMON] Reachy Mini daemon is already running (PID: {pid}). "
            "No need to spawn a new one."
        )
        return
    elif daemon_is_running and sim != use_sim:
        logger.info(
            f"[DAEMON] Reachy Mini daemon is already running (PID: {pid}) \
            with a different configuration. "
        )
        logger.info("[DAEMON] Killing the existing daemon...")
        assert pid is not None, "PID should not be None if daemon is running"
        os.kill(pid, 9)
        time.sleep(1)

    logger.info("[DAEMON] Starting a new daemon...")
    command = ["reachy-mini-daemon"]
    if use_sim:
        command.append("--sim")
    if hardware_config_filepath:
        command.extend(["--hardware-config-filepath", hardware_config_filepath])
    subprocess.Popen(command, start_new_session=True)

    # Wait for the daemon to initialize and start listening on Zenoh port
    logger.info("[DAEMON] Waiting for daemon to initialize...")
    time.sleep(wait_time)


def config_respeaker(respeaker_parameters: dict[str, Any], logger: Logger) -> None:
    """Configure the ReSpeaker microphone."""

    # If no parameters are provided, don't configure the ReSpeaker microphone
    if not respeaker_parameters:
        return

    respeaker = init_respeaker_usb()

    if respeaker is None:
        logger.warning(
            "No ReSpeaker device found. If you don't have Reachy Mini robot connected \
            to your machine, you can ignore this warning."
        )
        return

    for command, values in respeaker_parameters.items():
        # NOTE: code from reachy_mini.media.audio_control_utils.main()
        try:
            if RESPEAKER_PARAMETERS[command][3] == "ro":
                logger.error(f"Error: {command} is read-only and cannot be written to")
                continue

            if (
                RESPEAKER_PARAMETERS[command][4] != "float"
                and RESPEAKER_PARAMETERS[command][4] != "radians"
            ):
                values = [int(v) for v in values]

            if RESPEAKER_PARAMETERS[command][2] != len(values):
                logger.error(
                    f"Error: {command} value count is \
                    {RESPEAKER_PARAMETERS[command][2]}, \
                    but {len(values)} values provided"
                )
                continue

            logger.info(f"Writing to {command} with values: {values}")
            respeaker.write(command, values)
            time.sleep(0.1)
            logger.info(f"Write operation completed successfully for {command}")

        except Exception as e:
            error_msg = f"Error executing command {command}: {e}"
            logger.error(error_msg)

            # Check if it's a permission error, so far only seen on Linux
            if (
                "Errno 13" in str(e)
                or "Access denied" in str(e)
                or "insufficient permissions" in str(e)
            ):
                logger.error(
                    "This looks like a permissions error. You are most likely "
                    "on Linux and need to adjust udev rules for USB permissions."
                )
            respeaker.close()
            raise

    respeaker.close()
