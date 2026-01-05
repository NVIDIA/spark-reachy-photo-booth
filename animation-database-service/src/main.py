# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import pathlib
import time
from collections.abc import AsyncGenerator
from typing import override

import pydub
from audio.audio_conversion import convert_audio
from cache import LRUCacheDict
from configuration import DatabaseConfig
from models import Animation
from procedural.procedural_animations import look_at_animation, track_animation
from procedural.procedural_audio import generate_body_angle_sound
from utils import Degrees, RobotPosition, TrackingData, convert_bits_per_sample
from workmesh.config import load_config
from workmesh.messages import Animation as AnimationProto
from workmesh.messages import (
    AnimationData,
    AnimationFrame,
    Audio,
    AudioProperties,
    ClipData,
    EulerAngle,
    PlayClip,
    ProceduralClip,
    ProceduralClipUpdate,
    ProceduralState,
    ProceduralType,
    TrackingStatus,
    UserDetection,
    UserTrackingStatus,
)
from workmesh.service import Service, produces, subscribe
from workmesh.service_executor import ServiceExecutor

from workmesh import (
    clip_data_topic,
    play_clip_topic,
    procedural_clip_topic,
    user_detection_topic,
    user_tracking_status_topic,
)


class AnimationDatabaseService(Service):
    def __init__(self, config: DatabaseConfig) -> None:
        super().__init__(config)
        self.config: DatabaseConfig = config

        # Convert audio files to WAV and config format
        convert_audio(self.config, self.logger)

        # Create clips cache
        self.clip_cache: LRUCacheDict = LRUCacheDict(maxsize=100)

        # User tracking
        self.user_tracking_event: asyncio.Event = asyncio.Event()
        self.user_tracking_lock: asyncio.Lock = asyncio.Lock()
        self.tracker_data: TrackingData | None = None

    @override
    async def stop(self) -> None:
        self.clip_cache.clear()
        await super().stop()

    #########################
    # Procedural animations
    #########################

    @subscribe(user_detection_topic)
    @produces(procedural_clip_topic)
    async def on_user_detection(
        self, message: UserDetection
    ) -> AsyncGenerator[ProceduralClip, None]:
        if self.user_tracking_event.is_set():
            async with self.user_tracking_lock:
                assert self.tracker_data is not None
                tracking_data = self.tracker_data

            # If the tracking data is paused, don't generate a new animation
            if tracking_data.paused:
                return

            # Generate TRACK animation
            # Validate marker_id and keypoints availability
            marker_id = message.marker_id
            n_skeletons = len(message.skeletons)
            if marker_id >= n_skeletons:
                self.logger.error(
                    f"marker_id {marker_id} out of bounds for {n_skeletons} skeletons"
                )
                return
            skeleton = message.skeletons[marker_id]
            if not skeleton.keypoints:
                self.logger.error(
                    f"No keypoints available for skeleton at marker_id {marker_id}"
                )
                return

            # Apply neck offset based on bounding box size
            # The larger the bounding box, the smaller the offset (user is closer)
            neck_offset_factor = self.config.neck_offset_factor
            neck_y_offset = (
                1 - message.bounding_boxes[marker_id].width
            ) * neck_offset_factor

            # Track position: nose x and offset y (approx. neck position)
            track_x = skeleton.keypoints[0].x
            track_y = skeleton.keypoints[0].y + neck_y_offset

            # Generate TRACK animation
            track_anim: RobotPosition | None = track_animation(
                tracking_data.robot_position,
                track_x,
                track_y,
                tracking_data.slow_mode_distance_threshold,
                tracking_data.fast_mode_distance_threshold,
                tracking_data.slow_speed,
                tracking_data.fast_speed,
            )

            # Update user tracking status
            user_centered: bool = track_anim is None
            if tracking_data.user_centered != user_centered:
                await self.publish(
                    user_tracking_status_topic,
                    UserTrackingStatus(
                        action_uuid=tracking_data.action_uuid,
                        robot_id=message.robot_id,
                        status=TrackingStatus.USER_CENTERED
                        if user_centered
                        else TrackingStatus.USER_NOT_CENTERED,
                    ),
                )

            # Update tracking data
            async with self.user_tracking_lock:
                self.tracker_data.user_centered = user_centered
                if track_anim is not None:
                    self.tracker_data.robot_position = track_anim

            # If the user is centered, don't send a tracking animation update
            if user_centered:
                self.logger.debug(
                    "User is too close to the camera center. "
                    "Not sending tracking animation update."
                )
                return

            # Send proto message
            animation = AnimationProto(
                frame_rate=self.config.frame_rate,
                data=AnimationData(
                    frames=[
                        AnimationFrame(
                            head_rotation=EulerAngle(
                                roll=track_anim.head_rotation.roll,
                                pitch=track_anim.head_rotation.pitch,
                                yaw=track_anim.head_rotation.yaw,
                            ),
                            body_angle=track_anim.body_angle,
                        )
                    ],
                ),
            )

            clip_data = ClipData(
                action_uuid=tracking_data.action_uuid,
                robot_id=message.robot_id,
                animation=animation,
            )

            yield ProceduralClip(
                action_uuid=tracking_data.action_uuid,
                robot_id=message.robot_id,
                timestamp=int(time.time() * 1000),
                type=ProceduralType.TRACK,
                state=ProceduralState.UPDATE,
                update=ProceduralClipUpdate(clip=clip_data),
            )

    @subscribe(procedural_clip_topic)
    @produces(procedural_clip_topic)
    async def on_procedural_clip(
        self, message: ProceduralClip
    ) -> AsyncGenerator[ProceduralClip, None]:
        if (
            message.type == ProceduralType.LOOK_AT
            and message.state == ProceduralState.START
        ):
            # Error handling
            if not message.HasField("start"):
                self.logger.error("Start message for LOOK_AT is missing START action.")
                return
            if not message.start.HasField("look_at_parameters"):
                self.logger.error("LOOK_AT parameters are missing.")
                return
            if message.start.look_at_parameters.duration <= 0:
                self.logger.error("LOOK_AT duration is less than or equal to 0.")
                return

            self.logger.info("Received LOOK_AT START message.")

            # Get target
            if message.start.look_at_parameters.HasField("target_position"):
                target = message.start.look_at_parameters.target_position
            elif message.start.look_at_parameters.HasField("target_body_angle"):
                target = message.start.look_at_parameters.target_body_angle
            else:
                self.logger.error("LOOK_AT target is missing.")
                return

            volume = message.start.volume if message.start.HasField("volume") else 1.0

            # Generate LOOK_AT animation
            look_at_anim = look_at_animation(
                message.start.look_at_parameters.start_body_angle,
                target,
                message.start.look_at_parameters.duration,
                self.config.frame_rate,
            )

            # Generate procedural audio for the rotation
            audio = generate_body_angle_sound(
                duration=message.start.look_at_parameters.duration,
                volume=volume,
            )
            resampled_audio = convert_bits_per_sample(
                audio, self.config.audio_config.bits_per_sample
            )

            # Create animation data
            animation = AnimationProto(
                frame_rate=self.config.frame_rate,
                data=AnimationData(
                    frames=[AnimationFrame(body_angle=angle) for angle in look_at_anim]
                ),
            )

            # Create audio data
            audio = Audio(
                sample_rate=self.config.audio_config.sample_rate,
                bits_per_sample=self.config.audio_config.bits_per_sample,
                channel_count=self.config.audio_config.channel_count,
                audio_buffer=resampled_audio.tobytes(),
            )

            # Create clip data with both animation and audio
            clip_data = ClipData(
                action_uuid=message.action_uuid,
                robot_id=message.robot_id,
                animation=animation,
                audio=audio,
                audio_properties=AudioProperties(volume=volume),
            )

            # Send proto message with ClipData
            yield ProceduralClip(
                action_uuid=message.action_uuid,
                robot_id=message.robot_id,
                timestamp=int(time.time() * 1000),
                type=ProceduralType.LOOK_AT,
                state=ProceduralState.UPDATE,
                update=ProceduralClipUpdate(clip=clip_data),
            )

            self.logger.info(
                f"Sent look_at animation with audio: "
                f"{message.start.look_at_parameters.start_body_angle:.1f}° → {look_at_anim[-1]:.1f}° "  # noqa: E501
                f"over {message.start.look_at_parameters.duration}s"
            )

        elif message.type == ProceduralType.TRACK:
            if message.state == ProceduralState.START:
                # Error handling
                if not message.HasField("start"):
                    self.logger.error(
                        "Start message for TRACK is missing START action."
                    )
                    return
                if not message.start.HasField("track_parameters"):
                    self.logger.error("TRACK parameters are missing.")
                    return

                self.logger.info("Received TRACK START message.")

                # Set user tracking data
                async with self.user_tracking_lock:
                    self.tracker_data = TrackingData(
                        action_uuid=message.action_uuid,
                        slow_mode_distance_threshold=message.start.track_parameters.slow_mode_distance_threshold,
                        fast_mode_distance_threshold=message.start.track_parameters.fast_mode_distance_threshold,
                        slow_speed=message.start.track_parameters.slow_speed,
                        fast_speed=message.start.track_parameters.fast_speed,
                        robot_position=RobotPosition(),
                    )

                # Activate user tracking
                self.user_tracking_event.set()

            elif message.state == ProceduralState.STOP:
                self.logger.info("Received TRACK STOP message.")

                # Deactivate user tracking
                self.user_tracking_event.clear()
                async with self.user_tracking_lock:
                    self.tracker_data = None

            elif message.state == ProceduralState.PAUSE:
                self.logger.info("Received TRACK PAUSE message.")

                if not message.HasField("pause"):
                    self.logger.error("PAUSE message for TRACK is missing PAUSE action.")  # noqa: E501 # fmt: skip
                    return

                # Pause user tracking
                async with self.user_tracking_lock:
                    if self.tracker_data is None:
                        self.logger.warning("PAUSE message received for a clip that is not running. Ignoring.")  # noqa: E501 # fmt: skip
                        return

                    self.tracker_data.paused = message.pause.enable

    #########################
    # Retrieve clip data
    #########################

    @subscribe(play_clip_topic)
    @produces(clip_data_topic)
    async def on_play_clip(self, message: PlayClip):
        # Extract audio and animation data from clip
        animation: AnimationProto | None = None
        audio: Audio | None = None

        try:
            animation, audio = self._extract_data_from_clip(message.clip_name)
        except Exception as e:
            self.logger.exception(f"Error extracting data from clip: {e}")
            return

        # If no data is found, return
        if animation is None and audio is None:
            return

        # Send retrieved clip to the Composer
        yield ClipData(
            action_uuid=message.action_uuid,
            robot_id=message.robot_id,
            animation=animation,
            audio=audio,
            clip_properties=message.clip_properties,
            animation_properties=message.animation_properties,
            audio_properties=message.audio_properties,
        )

        self.logger.info(
            f"Sent clip '{message.clip_name}' with action uuid {message.action_uuid}."
        )

    def _extract_data_from_clip(
        self, clip_name: str
    ) -> tuple[AnimationProto | None, Audio | None]:
        """Extract animation and audio data from clip."""

        # Check if clip is in cache
        if clip_name in self.clip_cache:
            self.logger.debug(f"Clip '{clip_name}' found in cache.")
            return self.clip_cache[clip_name]

        # Initialize animation and audio data
        animation_data: AnimationProto | None = None
        audio_data: Audio | None = None

        animation_filename: pathlib.Path = (
            self.config.clip_directory / clip_name / f"{clip_name}.json"
        )
        audio_filename: pathlib.Path = (
            self.config.clip_directory / clip_name / f"{clip_name}_converted.wav"
        )

        # Check if clip exists
        if not animation_filename.exists() and not audio_filename.exists():
            self.logger.error(f"Clip '{clip_name}' not found.")
            return None, None

        # Extract animation data
        if animation_filename.exists():
            try:
                animation_json = Animation.from_json_file(animation_filename)
                animation_frames = self._create_animation_frames(animation_json)
                animation_data = AnimationProto(
                    data=AnimationData(frames=animation_frames),
                    frame_rate=animation_json.frame_rate,
                )
            except Exception as e:
                self.logger.exception(f"Error loading animation data from file: {e}")

        # Extract audio data
        if audio_filename.exists():
            try:
                audio: pydub.AudioSegment = pydub.AudioSegment.from_file(audio_filename)

                audio_data = Audio(
                    audio_buffer=audio.raw_data,
                    sample_rate=self.config.audio_config.sample_rate,
                    bits_per_sample=self.config.audio_config.bits_per_sample,
                    channel_count=self.config.audio_config.channel_count,
                )
            except Exception as e:
                self.logger.exception(f"Error loading audio data from file: {e}")

        # Cache clip data
        self.logger.debug(f"Caching clip '{clip_name}'.")
        self.clip_cache[clip_name] = (animation_data, audio_data)

        return animation_data, audio_data

    def _wrap_angle_to_180(self, angle: Degrees) -> Degrees:
        # NOTE: Reachy's antennas and base cannot rotate endlessly in one direction.
        # All rotation angles must remain within the [-180, 180] degree range.
        # Direct transitions between +180 and -180 degrees are not possible,
        # so values must be wrapped to prevent sudden jumps.

        # Wrap angle to be within [-180, 180]
        angle = ((angle + 180) % 360) - 180
        return angle

    def _create_animation_frames(
        self, animation_data: Animation
    ) -> list[AnimationFrame]:
        frames: list[AnimationFrame] = []

        for i in range(animation_data.data.n_frames):
            r_antenna_angle: Degrees | None = None
            l_antenna_angle: Degrees | None = None
            body_angle: Degrees | None = None
            head_rotation: EulerAngle | None = None
            head_position: list[float | None] = [None, None, None]

            # Antennas
            if animation_data.data.r_antenna_angle is not None:
                r_antenna_angle = self._wrap_angle_to_180(
                    animation_data.data.r_antenna_angle.frames[i]
                )
            if animation_data.data.l_antenna_angle is not None:
                l_antenna_angle = self._wrap_angle_to_180(
                    animation_data.data.l_antenna_angle.frames[i]
                )

            # Body rotation
            if animation_data.data.body_angle is not None:
                body_angle = self._wrap_angle_to_180(
                    animation_data.data.body_angle.frames[i]
                )

            # Head rotation
            if animation_data.data.head_rotation is not None:
                idx_roll = animation_data.data.head_rotation.joints.index("neck_roll")
                idx_pitch = animation_data.data.head_rotation.joints.index("neck_pitch")
                idx_yaw = animation_data.data.head_rotation.joints.index("neck_yaw")

                head_rotation = EulerAngle(
                    roll=animation_data.data.head_rotation.frames[i][idx_roll],
                    pitch=animation_data.data.head_rotation.frames[i][idx_pitch],
                    yaw=animation_data.data.head_rotation.frames[i][idx_yaw],
                )

            # Head position
            if animation_data.data.head_position is not None:
                joint_map = {
                    name: idx
                    for idx, name in enumerate(animation_data.data.head_position.joints)
                }
                coords = ["x", "y", "z"]
                for j, coord in enumerate(coords):
                    joint_name = f"head_{coord}"
                    if joint_name in joint_map:
                        idx = joint_map[joint_name]
                        head_position[j] = animation_data.data.head_position.frames[i][idx]  # noqa: E501 # fmt: skip

            frames.append(
                AnimationFrame(
                    r_antenna_angle=r_antenna_angle,
                    l_antenna_angle=l_antenna_angle,
                    head_rotation=head_rotation,
                    body_angle=body_angle,
                    head_position_x=head_position[0],
                    head_position_y=head_position[1],
                    head_position_z=head_position[2],
                )
            )

        return frames


async def main():
    config = load_config(DatabaseConfig)
    await ServiceExecutor([AnimationDatabaseService(config)]).run()


if __name__ == "__main__":
    asyncio.run(main())
