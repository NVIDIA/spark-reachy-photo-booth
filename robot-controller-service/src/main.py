# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any, override

import numpy as np
from configuration import RobotControllerConfig
from reachy_mini.reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from scipy.spatial.transform import Rotation as R
from utils import (
    clip_frame_by_max_delta,
    config_respeaker,
    remap_head_translation,
    start_daemon,
)
from workmesh.messages import AnimationFrame, EulerAngle, RealtimeDoA, RobotFrame
from workmesh.service import subscribe
from workmesh.service_executor import ServiceExecutor

from workmesh import (
    Service,
    load_config,
    produces,
    realtime_doa_topic,
    robot_frame_topic,
)


class RobotService(Service):
    def __init__(self, config: RobotControllerConfig) -> None:
        super().__init__(config)

        self.config = config
        self.robot_id = config.robot_id

        # Start time
        self.start_time: int = int(time.time() * 1000)

        # Configure ReSpeaker microphone
        config_respeaker(
            self.config.respeaker_config.microphone_parameters, self.logger
        )

        # TODO: change it when Pollen adds a way to use hardware config
        # directly in the ReachyMini class
        start_daemon(
            logger=self.logger,
            use_sim=self.config.reachy_config.use_sim,
            hardware_config_filepath=self.config.reachy_config.hardware_config_filepath,
            wait_time=self.config.reachy_config.daemon_wait_time,
        )

        # Connect to Reachy
        self.reachy = ReachyMini(
            spawn_daemon=False,
            localhost_only=self.config.reachy_config.localhost_only,
            use_sim=self.config.reachy_config.use_sim,
            timeout=self.config.reachy_config.timeout,
            log_level=self.config.log_level,
            media_backend=self.config.reachy_config.media_backend.value,
        )

        self.logger.info("Connected to Reachy Mini!")

        # Start the doa task
        self.doa_task: asyncio.Task | None = None
        self.stop_event: asyncio.Event = asyncio.Event()
        if self.config.respeaker_config.enable_doa:
            self.doa_task = self.create_task(self.send_doa())  # type: ignore

        # Frame rate tracking
        self.frame_timestamps: list[float] = []
        self.last_framerate_log_time: float = time.time()

    @override
    async def stop(self) -> None:
        # Stop service tasks
        self.stop_event.set()
        if self.doa_task is not None:
            await self.doa_task

        # Go to zero pose
        await self._go_to_zero_pose(2.0)

        # Disconnect from Reachy
        self.reachy.client.disconnect()
        self.logger.info("Disconnected from Reachy Mini!")

        # Stop the service
        await super().stop()

    async def _go_to_zero_pose(self, duration: float = 2.0) -> None:
        self.logger.info("Going to zero pose...")
        self.reachy.goto_target(
            head=np.eye(4), antennas=[0, 0], body_yaw=0, duration=duration
        )
        await asyncio.sleep(duration)

    def _get_current_position(self) -> AnimationFrame:
        current_position = self.reachy.get_current_joint_positions()
        head_position = self.reachy.get_current_head_pose()
        head_position[:3, 3] *= 100  # Convert from m to cm

        # Extract head rotation
        rot = R.from_matrix(head_position[:3, :3])
        head_roll, head_pitch, head_yaw = rot.as_euler("xyz", degrees=True)

        # Extract translation
        translation = head_position[:, 3]
        head_position_x = translation[0]
        head_position_y = translation[1]
        head_position_z = translation[2]

        return AnimationFrame(
            r_antenna_angle=np.rad2deg(current_position[1][0]),
            l_antenna_angle=np.rad2deg(current_position[1][1]),
            body_angle=np.rad2deg(current_position[0][0]),
            head_position_x=head_position_x,
            head_position_y=head_position_y,
            head_position_z=head_position_z,
            head_rotation=EulerAngle(
                roll=head_roll,
                pitch=head_pitch,
                yaw=head_yaw,
            ),
        )

    @produces(realtime_doa_topic)
    async def send_doa(self) -> AsyncGenerator[RealtimeDoA, Any]:
        self.logger.info("Retrieving DoA started!")
        while not self.stop_event.is_set():
            doa = None
            try:
                if self.reachy.media.audio is not None:
                    doa = self.reachy.media.audio.get_DoA()

                if doa is not None:
                    angle = np.rad2deg(doa[0]) - 90  # -90: left, 0: front, 90: right
                    self.logger.debug(f"[DoA]: {angle}° Speech detected {doa[1]}")
                    yield RealtimeDoA(angle=angle, speech_detected=doa[1])
            except Exception as e:
                self.logger.error(f"Error getting DoA from audio hardware: {e}")

            await asyncio.sleep(self.config.respeaker_config.doa_interval)
        self.logger.info("Retrieving DoA stopped!")

    @subscribe(robot_frame_topic)
    async def on_robot_frame_topic(self, message: RobotFrame):
        # Ignore all messages if the service is stopping
        if self.stop_event.is_set():
            return

        if message.timestamp < self.start_time:
            self.logger.debug("Skipping message. Message is older than the service.")
            return

        # Check if the message is for this robot
        if message.robot_id != self.robot_id.to_proto():
            return

        # Track frame rate
        current_time = time.time()
        self.frame_timestamps.append(current_time)

        # Log average frame rate every X seconds
        time_since_last_log = current_time - self.last_framerate_log_time
        if time_since_last_log >= self.config.frame_rate_log_interval:
            if len(self.frame_timestamps) > 1:
                time_span = self.frame_timestamps[-1] - self.frame_timestamps[0]
                if time_span > 0:
                    avg_fps = (len(self.frame_timestamps) - 1) / time_span
                    self.logger.debug(
                        f"Average frame rate: {avg_fps:.2f} FPS (over {time_span:.1f}s,"
                        f" {len(self.frame_timestamps)} frames)"
                    )

            # Reset tracking
            self.frame_timestamps.clear()
            self.last_framerate_log_time = current_time

        # Clip frame by max delta
        current_position = self._get_current_position()
        clipped_frame, was_clipped = clip_frame_by_max_delta(
            message.frame,
            current_position,
            self.config.max_delta_per_frame,
        )

        if was_clipped:
            self.logger.warning(
                "Frame was clipped due to big delta."
                + f"\nCurrent frame: {current_position}"
                + f"\nMessage frame: {message.frame}"
                + f"\nClipped frame: {clipped_frame}"
            )

        # Remap head translation
        head_position = remap_head_translation(
            clipped_frame, scale=(10, 10, 10), translation_offset=(0.0, 0.0, 0.0)
        )

        # Set the position
        body_yaw = np.deg2rad(clipped_frame.body_angle)
        antennas = np.deg2rad(
            [clipped_frame.r_antenna_angle, clipped_frame.l_antenna_angle]
        )
        head = create_head_pose(
            x=head_position[0],
            y=head_position[1],
            z=head_position[2],
            roll=clipped_frame.head_rotation.roll,
            pitch=clipped_frame.head_rotation.pitch,
            yaw=clipped_frame.head_rotation.yaw,
            degrees=True,
            mm=True,
        )

        self.logger.debug(f"Current position: {current_position}")
        self.logger.debug(f"Message position: {message.frame}")
        self.logger.debug(f"Final position: {clipped_frame}")
        self.reachy.set_target(head=head, antennas=antennas, body_yaw=body_yaw)


async def main():
    config = load_config(RobotControllerConfig)
    await ServiceExecutor([RobotService(config)]).run()


if __name__ == "__main__":
    asyncio.run(main())
