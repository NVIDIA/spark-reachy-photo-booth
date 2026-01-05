# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import heapq
import time
from collections.abc import AsyncGenerator
from typing import Any, override

import pydub
from audio_mixer import AudioMixer
from clips.animations import (
    JOINT_NAMES,
    Degrees,
    EulerAngles,
    Frame,
    blend_multiple_frames,
)
from clips.base_clips import AudioClip, BlendingAnimationClip
from clips.procedural_clips import (
    LookAtAnimation,
    ProceduralClip,
    ProceduralType,
    TrackAnimation,
)
from clips.static_clips import StaticAnimation, StaticClip
from configuration import CompositorConfig, Robot
from utils import create_animation_frames, create_robot_message, validate_audio_settings
from workmesh.config import load_config
from workmesh.messages import (
    Audio,
    ChangeVolume,
    ClipData,
    ClipStatus,
    Command,
    ProceduralClipStop,
    ProceduralState,
    RobotFrame,
    ServiceCommand,
    StopClip,
)
from workmesh.messages import ProceduralClip as ProceduralClipMessage
from workmesh.messages import Service as ServiceName
from workmesh.service import Service, produces, subscribe
from workmesh.service_executor import ServiceExecutor

from workmesh import (
    change_volume_topic,
    clip_data_topic,
    clip_status_topic,
    procedural_clip_topic,
    robot_frame_topic,
    service_command_topic,
    stop_clip_topic,
)


class AnimationCompositorService(Service):
    def __init__(self, config: CompositorConfig) -> None:
        super().__init__(config)

        # Config
        self.config: CompositorConfig = config

        # Robot identification
        self.robot_id: Robot = self.config.robot_id
        self.logger.info(
            f"Animation Compositor initialized for robot: {self.config.robot_id.name}"
        )

        # Procedural animations
        self.procedural_clips: dict[ProceduralType, ProceduralClip] = {}
        self.static_body_angle: Degrees | None = None

        # Clips
        self.clip_lock: asyncio.Lock = asyncio.Lock()
        self.clip_list: dict[str, StaticClip] = {}
        self.priority_dict: dict[int, str] = {}

        # Audio
        self.audio_mixer: AudioMixer | None = None
        try:
            self.audio_mixer = AudioMixer(
                self.config.frame_rate, self.config.audio_config, self.logger
            )
        except Exception as e:
            self.logger.exception(f"Error when creating audio mixer: {e}")
            self.logger.warning("Audio playback not available. Can't use the Audio Mixer.")  # noqa: E501 # fmt: skip
            self.audio_mixer = None

        # Play frame
        self.create_task(self.send_animation_frame())  # type: ignore
        self.frame_stop_event: asyncio.Event = asyncio.Event()

        # Frame smoothing (delta clipping)
        self.previous_frame: Frame | None = None
        self.max_delta_per_frame: dict[str, float] = self.config.max_delta_per_frame

    @override
    async def stop(self) -> None:
        # Stop composer frame task first
        self.frame_stop_event.set()

        # Clear clip list
        async with self.clip_lock:
            self.clip_list.clear()
            self.priority_dict.clear()
            self.procedural_clips.clear()

        # Stop audio mixer
        if self.audio_mixer:
            await self.audio_mixer.close()

        self.logger.info("Animation compositor was stopped.")
        await super().stop()

    @subscribe(service_command_topic)
    async def on_service_command(self, message: ServiceCommand):
        if (
            message.target_service == ServiceName.COMPOSITOR
            and message.command == Command.RESTART
        ):
            self.logger.info("---- RESTARTING ----")
            for clip in self.clip_list.values():
                await self.on_stop_clip(StopClip(action_uuid=clip.action_uuid))

            for procedural_type, clip in self.procedural_clips.items():
                await self.on_procedural_clip(
                    ProceduralClipMessage(
                        action_uuid=clip.action_uuid,
                        type=ProceduralType.to_proto(procedural_type),
                        state=ProceduralState.STOP,
                        stop=ProceduralClipStop(
                            blend_out_duration=0.5,
                        ),
                    )
                )

            self.logger.info("---- RESTARTED ----")

    #############################
    # Clip management
    ##############################

    @subscribe(clip_data_topic)
    async def on_play_clip(self, message: ClipData) -> None:
        audio_clip = None
        animation_clip = None
        duration = 0.0

        # Filter out clips for other robots
        if message.robot_id != self.robot_id.to_proto():
            return

        async with self.clip_lock:
            if message.action_uuid in self.clip_list:
                self.logger.warning(f"Clip {message.action_uuid} already exists.")
                return

        if message.HasField("animation"):
            animation = create_animation_frames(message.animation.data)

            duration = len(message.animation.data.frames) / message.animation.frame_rate
            animation_clip = StaticAnimation(
                frame_rate=message.animation.frame_rate,
                duration=duration,
                animation_data=animation,
            )

            if message.animation_properties.HasField("opacity"):
                animation_clip.opacity = message.animation_properties.opacity
            if message.animation_properties.HasField("priority"):
                animation_clip.priority = message.animation_properties.priority

            # Blend in/out effects
            if message.animation_properties.HasField("blending"):
                if message.animation_properties.blending.HasField("transition_in"):
                    animation_clip.blend_in_duration = (
                        message.animation_properties.blending.transition_in
                    )
                if message.animation_properties.blending.HasField("transition_out"):
                    animation_clip.blend_out_duration = (
                        message.animation_properties.blending.transition_out
                    )

        if message.HasField("audio") and self.audio_mixer:
            if not validate_audio_settings(message.audio, self.config.audio_config):
                self.logger.warning(
                    "Received audio doesn't match the audio mixer settings: "
                    + f"{message.audio.sample_rate} != {self.config.audio_config.sample_rate}, "  # noqa: E501
                    + f"{message.audio.bits_per_sample} != {self.config.audio_config.bits_per_sample}, "  # noqa: E501
                    + f"{message.audio.channel_count} != {self.config.audio_config.channel_count}"  # noqa: E501
                )
            else:
                audio_clip = AudioClip(
                    audio_data=pydub.AudioSegment(
                        message.audio.audio_buffer,
                        sample_width=message.audio.bits_per_sample // 8,
                        frame_rate=message.audio.sample_rate,
                        channels=message.audio.channel_count,
                    )
                )

                # Volume effect
                if (
                    message.audio_properties.HasField("volume")
                    and message.audio_properties.volume != 0
                ):
                    audio_clip.change_volume(message.audio_properties.volume)

                # Fade in/out effects
                fade_in = int(message.audio_properties.fading.transition_in * 1000)
                fade_out = int(message.audio_properties.fading.transition_out * 1000)

                if fade_in > 0:
                    audio_clip.audio_data = audio_clip.audio_data.fade_in(fade_in)
                    audio_clip.fade_in_duration = fade_in
                if fade_out > 0:
                    audio_clip.audio_data = audio_clip.audio_data.fade_out(fade_out)
                    audio_clip.fade_out_duration = fade_out

                # Duration
                duration = max(duration, audio_clip.audio_data.duration_seconds)

        if not animation_clip and not audio_clip:
            self.logger.warning(
                f"Can't create clip {message.action_uuid}. No animation or audio."
            )
            await self.send_clip_error(message.action_uuid)
            return

        # Create clip
        clip = StaticClip(
            action_uuid=message.action_uuid,
            audio=audio_clip,
            animation=animation_clip,
            duration=duration,
            loop=message.clip_properties.loop,
        )

        if message.clip_properties.HasField("loop_overlap"):
            clip.loop_overlap = message.clip_properties.loop_overlap

        async with self.clip_lock:
            self.clip_list[clip.action_uuid] = clip

            # Add clip to priority dictionary only if it has an animation
            if clip.animation:
                assert isinstance(clip.animation, StaticAnimation)
                old_clip_uuid: str | None = self.priority_dict.get(
                    clip.animation.priority
                )

                if old_clip_uuid is not None:
                    old_clip = self.clip_list[old_clip_uuid]
                    assert old_clip.animation is not None

                    # If clip has started, stop and transition out
                    if old_clip.start_time is not None:
                        self._stop_and_transition_out(
                            old_clip,
                            time.time(),
                            old_clip.animation.blend_out_duration,
                            0.0,
                        )
                    # If clip hasn't started, remove it from the clip list
                    else:
                        self.clip_list.pop(old_clip_uuid)
                        self.logger.info(f"Stopped clip {old_clip_uuid}.")

                    self.logger.debug(
                        f"Clip {old_clip_uuid} was replaced by "
                        + f"{clip.action_uuid} with same priority."
                    )

                self.priority_dict[clip.animation.priority] = clip.action_uuid

        self.logger.info(f"Added clip {clip.action_uuid} to the clip list.")

    @subscribe(stop_clip_topic)
    async def on_stop_clip(self, message: StopClip):
        current_time = time.time()

        async with self.clip_lock:
            clip = self.clip_list.get(message.action_uuid)

            # Check if clip exists
            if clip is None:
                self.logger.warning(
                    f"Can't stop clip {message.action_uuid}. The clip doesn't exist or has already finished."  # noqa: E501 #fmt: skip
                )
                return

            # Check if clip has started
            if not clip.start_time or not clip.end_time:
                self.logger.warning(
                    f"Can't stop clip {message.action_uuid}. It has not started yet."
                )
                return

            # Check if clip is already blending/fading out
            if not clip.loop and clip.in_transition_out(current_time - clip.start_time):
                self.logger.warning(
                    f"Can't stop clip {message.action_uuid}. It is already blending out."  # noqa: E501 #fmt: skip
                )
                return

            # Stop when clip is blending/fading in
            if clip.in_transition_in(current_time - clip.start_time):
                self.logger.warning(
                    f"Stopping clip {message.action_uuid} while blending/fading in."  # noqa: E501 #fmt: skip
                )

                animation_transition_out = 0.0
                audio_transition_out = 0.0

                # Stop and blend out animation
                if clip.animation:
                    in_transition_fraction = (
                        current_time - clip.start_time
                    ) / clip.animation.blend_in_duration
                    animation_transition_out = (
                        clip.animation.blend_out_duration * in_transition_fraction
                    )

                # Stop and fade out audio
                if clip.audio and (
                    message.fade_out > 0 or clip.audio.fade_out_duration > 0
                ):
                    in_transition_fraction = (
                        current_time - clip.start_time
                    ) / clip.audio.fade_in_duration
                    audio_transition_out = (
                        int(message.fade_out * 1000)
                        if message.fade_out > 0
                        else clip.audio.fade_out_duration
                    )

                self._stop_and_transition_out(
                    clip, current_time, animation_transition_out, audio_transition_out
                )

            # Normal stop
            else:
                animation_transition_out = 0.0
                audio_transition_out = 0.0

                # Stop and blend out animation
                if clip.animation:
                    animation_transition_out = clip.animation.blend_out_duration

                # Stop and fade out audio
                if clip.audio and (
                    message.fade_out > 0 or clip.audio.fade_out_duration > 0
                ):
                    audio_transition_out = (
                        int(message.fade_out * 1000)
                        if message.fade_out > 0
                        else clip.audio.fade_out_duration
                    )

                self._stop_and_transition_out(
                    clip, current_time, animation_transition_out, audio_transition_out
                )

    def _stop_and_transition_out(
        self,
        clip: StaticClip,
        current_time: float,
        animation_transition_out: float,
        audio_transition_out: float,
    ) -> None:
        clip.end_time = current_time

        assert clip.start_time is not None
        assert clip.end_time is not None
        assert isinstance(clip.animation, StaticAnimation)

        # Stop and blend out animation
        if clip.animation and animation_transition_out > 0:
            clip.animation.cut(
                clip.end_time - clip.start_time + animation_transition_out
            )

        # Stop and fade out audio
        if clip.audio and audio_transition_out > 0 and self.audio_mixer:
            self.audio_mixer.fade_out_clip(clip.action_uuid, audio_transition_out)

        clip.end_time += max(audio_transition_out, animation_transition_out)
        clip.duration = clip.end_time - clip.start_time
        clip.loop = False

        self.logger.debug(f"Stopping clip {clip.action_uuid}...")

    @subscribe(change_volume_topic)
    async def on_change_volume(self, message: ChangeVolume):
        if not self.audio_mixer:
            self.logger.warning("Can't change volume. Audio mixer not initialized.")
            return

        if self.audio_mixer.change_volume(message.action_uuid, message.volume):
            await self.send_clip_updated(message.action_uuid, time.time())

    #############################
    # Procedural animations
    ##############################

    @subscribe(procedural_clip_topic)
    async def on_procedural_clip(self, message: ProceduralClipMessage):
        # Filter out procedural clip for other robots
        if message.robot_id != self.robot_id.to_proto():
            return

        procedural_type: ProceduralType = ProceduralType.from_proto(message.type)

        # Start procedural clip
        if message.state == ProceduralState.START:
            if not message.HasField("start"):
                self.logger.error(
                    f"Procedural clip '{procedural_type.name}' "
                    + "missing start parameters."
                )
                return

            elapsed = 0.0
            old_frame: Frame | None = None
            async with self.clip_lock:
                # Check if procedural clip is already running
                if procedural_type in self.procedural_clips:
                    clip: ProceduralClip = self.procedural_clips[procedural_type]
                    current_time = time.time()
                    if clip.start_time is not None:
                        elapsed = current_time - clip.start_time
                        # Replace existing procedural clip if it is blending out
                        if clip.is_blending_out(elapsed):
                            self.logger.warning(
                                f"Procedural clip '{procedural_type.name}' is currently"
                                " blending out. A new instance will replace the"
                                " existing one."
                            )
                            old_frame = self.procedural_clips[procedural_type].blend_out(elapsed)  # noqa: E501 #fmt: skip

                        # Replace existing procedural clip if it has finished before
                        # being processed in the main loop
                        elif (
                            clip.animation
                            and clip.animation.duration is not None
                            and clip.animation.duration > 0
                            and elapsed > clip.animation.duration
                        ):
                            self.logger.warning(
                                f"Procedural clip '{procedural_type.name}' has finished"
                                " while receiving a new START message."
                                " A new instance will replace the existing one."
                            )

                            # Mark it as finished
                            await self.send_clip_finished(
                                self.procedural_clips[procedural_type].action_uuid,
                                current_time,
                            )

                            # Remove it from the list
                            self.procedural_clips.pop(procedural_type)
                            self.logger.info(
                                f"Procedural animation {procedural_type.name} finished."
                            )
                        # Ignore if it is playing and not finishing
                        else:
                            self.logger.warning(
                                f"Procedural clip '{procedural_type.name}' is already"
                                " active. Ignoring start request."
                            )
                            await self.send_clip_error(message.action_uuid)
                            return

                # Create procedural clip
                procedural_clip: ProceduralClip | None = None
                if procedural_type == ProceduralType.LOOK_AT:
                    procedural_clip = LookAtAnimation(action_uuid=message.action_uuid)  # noqa: E501 #fmt: skip
                elif procedural_type == ProceduralType.TRACK:
                    blend_in_duration = 0.0
                    if message.start.HasField("blend_in_duration"):
                        blend_in_duration = message.start.blend_in_duration

                    procedural_clip = TrackAnimation(
                        action_uuid=message.action_uuid,
                        blend_in_duration=blend_in_duration,
                    )

                    if old_frame is not None:
                        old_frame.clamp_frame(self.config.joint_limits)
                        procedural_clip.set_blend_in_frame(old_frame)
                        procedural_clip.add_frame(old_frame)

                if procedural_clip is not None:
                    self.procedural_clips[procedural_type] = procedural_clip

            self.logger.info(f"Started procedural clip {procedural_type.name}")

        # Update procedural clip with generated frames
        elif message.state == ProceduralState.UPDATE:
            if not message.HasField("update"):
                self.logger.error(
                    f"Procedural clip '{procedural_type.name}' "
                    + "missing update parameters."
                )
                return

            async with self.clip_lock:
                # Ignore if it isn't running, is blending out or is paused
                if (
                    procedural_type not in self.procedural_clips
                    or self.procedural_clips[procedural_type].end_time is not None
                    or self.procedural_clips[procedural_type].is_paused()
                ):
                    return

                # Type specific actions
                if procedural_type == ProceduralType.LOOK_AT:
                    # Handle ClipData with both animation and audio
                    clip_data: ClipData = message.update.clip

                    # LOOK_AT expects audio and animation
                    if not clip_data.HasField("animation"):
                        self.logger.error(
                            f"Procedural clip '{procedural_type.name}' "
                            + "missing animation."
                        )
                        return

                    # Validate audio
                    audio: Audio | None = None
                    if clip_data.HasField("audio") and self.audio_mixer:
                        audio = clip_data.audio

                    if audio is not None and not validate_audio_settings(
                        clip_data.audio, self.config.audio_config
                    ):
                        self.logger.warning(
                            "Received audio doesn't match the audio mixer settings: "
                            + f"{clip_data.audio.sample_rate} != {self.config.audio_config.sample_rate}, "  # noqa: E501
                            + f"{clip_data.audio.bits_per_sample} != {self.config.audio_config.bits_per_sample}, "  # noqa: E501
                            + f"{clip_data.audio.channel_count} != {self.config.audio_config.channel_count}"  # noqa: E501
                        )

                        # Ignore audio if it doesn't match the audio mixer settings
                        audio = None

                    volume = 1.0
                    if not self.audio_mixer:
                        volume = None
                    elif clip_data.audio_properties.HasField("volume"):
                        volume = clip_data.audio_properties.volume

                    # Add animation
                    self.procedural_clips[procedural_type].update_animation(
                        clip_data.animation.data.frames,
                        audio,
                        volume=volume,
                        duration=len(clip_data.animation.data.frames)
                        / clip_data.animation.frame_rate,
                    )

                    # Reset previous static body angle from previous LOOK_AT
                    self.static_body_angle = None

                elif procedural_type == ProceduralType.TRACK:
                    # Add animation
                    self.procedural_clips[procedural_type].update_animation(
                        message.update.clip.animation.data.frames
                    )

            self.logger.debug(f"Updated procedural clip {procedural_type.name}.")

        # Stop procedural clip manually
        elif message.state == ProceduralState.STOP:
            if not message.HasField("stop"):
                self.logger.error(
                    f"Procedural clip '{procedural_type.name}' "
                    + "missing stop parameters."
                )
                return

            # Ignore if it isn't running
            if procedural_type not in self.procedural_clips:
                self.logger.warning(
                    f"Can't stop procedural clip '{procedural_type.name}'. "
                    + "It is not running."
                )
                return

            # Ignore if it has already stopped
            end_time = self.procedural_clips[procedural_type].end_time
            if end_time is not None and time.time() > end_time:
                self.logger.warning(
                    f"Can't stop procedural clip '{procedural_type.name}'. "
                    + "It has already stopped."
                )
                return

            # Stop procedural clip
            async with self.clip_lock:
                procedural_clip = self.procedural_clips[procedural_type]

                if procedural_type == ProceduralType.TRACK:
                    blend_out_duration = 0.0
                    if message.stop.HasField("blend_out_duration"):
                        blend_out_duration = message.stop.blend_out_duration

                    assert isinstance(procedural_clip, TrackAnimation)
                    procedural_clip.stop(blend_out_duration)

                    animation_frame = procedural_clip.get_frame_by_index(-1)

                    # Save the body angle
                    if animation_frame is not None:
                        assert animation_frame.body_angle is not None
                        if self.static_body_angle is None:
                            self.static_body_angle = 0.0
                        self.static_body_angle += animation_frame.body_angle

                        assert procedural_clip.animation is not None
                        assert procedural_clip.animation.animation_data is not None
                        assert (
                            procedural_clip.animation.animation_data.frames is not None
                        )
                        procedural_clip.animation.animation_data.frames[
                            0
                        ].body_angle = None

                elif procedural_type == ProceduralType.LOOK_AT:
                    blend_out_duration = 0
                    if message.stop.HasField("blend_out_duration"):
                        blend_out_duration = message.stop.blend_out_duration
                    procedural_clip.stop(0)

            self.logger.info(f"Stopped procedural clip {procedural_type.name}")

        elif message.state == ProceduralState.PAUSE:
            if not message.HasField("pause"):
                self.logger.error(
                    f"Procedural clip '{procedural_type.name}' "
                    + "missing pause parameters."
                )
                return

            async with self.clip_lock:
                if (
                    procedural_type not in self.procedural_clips
                    or self.procedural_clips[procedural_type].start_time is None
                ):
                    self.logger.warning(
                        "Can't pause or resume procedural clip"
                        + f" '{procedural_type.name}'. It is not running."
                    )
                    return

                if message.pause.enable == self.procedural_clips[procedural_type].is_paused():  # noqa: E501 #fmt: skip
                    action = "paused" if message.pause.enable else "playing"
                    self.logger.warning(f"Procedural clip '{procedural_type.name}' is already {action}.")  # noqa: E501 #fmt: skip
                    return

                self.procedural_clips[procedural_type].pause(message.pause.enable)

            action = "Paused" if message.pause.enable else "Resumed"
            self.logger.info(f"{action} procedural clip {procedural_type.name}")

    #############################
    # Frame generation
    ##############################

    @produces(robot_frame_topic)
    async def send_animation_frame(self) -> AsyncGenerator[RobotFrame, Any]:
        self.logger.info("Animation compositor play loop started!")

        while not self.frame_stop_event.is_set():
            current_time = time.time()
            try:
                blending_clips: list[BlendingAnimationClip] = []

                async with self.clip_lock:
                    removed_clips: list[str] = []

                    # Process clips
                    for action_uuid, clip in self.clip_list.items():
                        # Start clip
                        if not clip.start_time:
                            clip.start_time = current_time
                            clip.end_time = current_time + clip.duration

                            if clip.audio and self.audio_mixer:
                                await self.audio_mixer.play(clip)

                            # Publish clip started
                            await self.send_clip_started(action_uuid, current_time)
                            self.logger.info(f"Started clip {action_uuid}.")

                        # Stop clip or loop if needed
                        assert clip.start_time and clip.end_time
                        if current_time > clip.end_time:
                            if not clip.loop:
                                # Stop audio in the audio mixer
                                if clip.audio and self.audio_mixer:
                                    self.audio_mixer.stop(clip.action_uuid)

                                # Remove the priority mapping if this clip
                                # is the current one for its priority
                                if clip.animation:
                                    assert isinstance(clip.animation, StaticAnimation)
                                    current_uuid = self.priority_dict.get(clip.animation.priority)  # noqa: E501 #fmt: skip
                                    if current_uuid == action_uuid:
                                        self.priority_dict.pop(clip.animation.priority)

                                # Remove clip from compositor
                                removed_clips.append(action_uuid)
                                continue

                            # Loop clip
                            clip.start_time = current_time
                            clip.end_time = current_time + clip.duration
                            self.logger.debug(f"Looped clip {action_uuid}.")

                        # Create blending clip
                        if (
                            clip.animation
                            and clip.start_time <= current_time <= clip.end_time
                        ):
                            clip_time = current_time - clip.start_time

                            assert isinstance(clip.animation, StaticAnimation)
                            assert clip.animation.duration is not None
                            assert clip.animation.animation_data is not None

                            # Blend in
                            weight_in = 1.0
                            if clip.animation.is_blending_in(clip_time):
                                weight_in = clip_time / clip.animation.blend_in_duration

                            # Avoid blending in again in loops
                            if (
                                not clip.animation.finished_blend_in
                                and weight_in >= 1.0
                            ):
                                clip.animation.finished_blend_in = True

                            # Blend out
                            weight_out = 1.0
                            if not clip.loop and clip.animation.is_blending_out(
                                clip_time
                            ):
                                weight_out = (
                                    clip.animation.duration - clip_time
                                ) / clip.animation.blend_out_duration

                            # Combine weight with opacity
                            weight = min(weight_out, weight_in) * clip.animation.opacity

                            # Add blending clip
                            if weight > 0:
                                # Evaluate frame at current time
                                frame = clip.animation.animation_data.eval_frame_loop(
                                    clip.animation.duration,
                                    clip_time,
                                    clip.loop,
                                    clip.loop_overlap,
                                )

                                # Remove reference pose offsets
                                frame = frame.subtract_frame(self.config.offset_pose)  # noqa: E501 #fmt: skip

                                # Add frame to blending clips
                                blending_clips.append(
                                    BlendingAnimationClip(
                                        priority=clip.animation.priority,
                                        frame=frame,
                                        weight=frame.weight_mask * weight,
                                    )
                                )

                    # Remove clips
                    for action_uuid in removed_clips:
                        self.clip_list.pop(action_uuid)

                        # Publish clip finished
                        await self.send_clip_finished(action_uuid, current_time)
                        self.logger.info(f"Clip {action_uuid} finished.")

                # Compose frame
                composed_frame = self._compose_animation(blending_clips)

                # Process procedural animations
                composed_frame = await self._process_procedural_clips(
                    current_time, composed_frame
                )

                # Apply yaw to head yaw
                if composed_frame.body_angle is not None:
                    composed_frame = composed_frame.additive_blend(
                        Frame(head_rotation=EulerAngles(yaw=composed_frame.body_angle))
                    )

                # Apply reference pose offsets
                composed_frame = composed_frame.additive_blend(self.config.offset_pose)  # noqa: E501 #fmt: skip

                # Clamp frame to joint limits
                composed_frame.clamp_frame(self.config.joint_limits)

                # Fill missing joints with reference pose
                composed_frame = composed_frame.fill_missing_joints()

                # Smooth frame transitions to prevent sudden jumps
                smoothed_frame: Frame = composed_frame
                if self.previous_frame is not None:
                    smoothed_frame, was_clipped = (
                        composed_frame.clip_frame_by_max_delta(
                            self.previous_frame, self.max_delta_per_frame
                        )
                    )

                    if was_clipped:
                        self.logger.warning(
                            "Frame was smoothed due to big delta."
                            + f"\nPrevious frame: {self.previous_frame}"
                            + f"\nComposed frame: {composed_frame}"
                            + f"\nSmoothed frame: {smoothed_frame}"
                        )

                # Update previous frame for next iteration
                self.previous_frame = smoothed_frame

                # Send frame
                self.logger.debug(f"Frame: {smoothed_frame}")
                yield create_robot_message(self.robot_id, smoothed_frame, current_time)

                # Wait for next frame
                await asyncio.sleep(
                    1 / self.config.frame_rate - (time.time() - current_time)
                )
            except Exception as e:
                self.logger.exception(f"Error sending frame: {e}")
            finally:
                # Wait for next frame
                await asyncio.sleep(
                    1 / self.config.frame_rate - (time.time() - current_time)
                )

        self.logger.info("Animation compositor play loop stopped!")

    def _compose_animation(self, blending_clips: list[BlendingAnimationClip]) -> Frame:
        frames: list[Frame] = []
        weights: list[dict[str, float]] = []
        remaining_weight = {joint: 1.0 for joint in JOINT_NAMES}

        # No blending clips, return reference pose
        if len(blending_clips) <= 0:
            return Frame.reference_pose()

        # Calculate final blending weights
        heapq.heapify(blending_clips)  # Sort blending clips by priority
        while blending_clips:
            if sum(remaining_weight.values()) <= 0:
                break

            blending_clip = heapq.heappop(blending_clips)
            weight: dict[str, float] = {joint: 0.0 for joint in JOINT_NAMES}

            # Calculate joint weight
            for i, joint in enumerate(JOINT_NAMES):
                if remaining_weight[joint] > 0:
                    w = min(blending_clip.weight[i], remaining_weight[joint])
                    weight[joint] = w
                    remaining_weight[joint] = remaining_weight[joint] - w

            # Add frame to blend only if there is any weight
            if sum(weight.values()) > 0:
                frames.append(blending_clip.frame)
                weights.append(weight)

        # Blend frames
        total_weights = {joint: 1.0 - remaining_weight[joint] for joint in JOINT_NAMES}  # noqa: E501 #fmt: skip
        return blend_multiple_frames(frames, weights, total_weights)

    async def _process_procedural_clips(
        self, current_time: float, composed_frame: Frame
    ) -> Frame:
        new_composed_frame: Frame = composed_frame
        stop_procedural: list[ProceduralType] = []
        async with self.clip_lock:
            # Process procedural clips
            for procedural_type, procedural_clip in self.procedural_clips.items():
                if (
                    procedural_clip.animation is not None
                    and procedural_clip.animation.animation_data is None
                ):
                    # Ignore if animation data hasn't been provided yet
                    continue

                # Start procedural clip
                if procedural_clip.start_time is None:
                    procedural_clip.start(current_time)
                    await self.send_clip_started(
                        procedural_clip.action_uuid, current_time
                    )

                    if procedural_clip.audio is not None and self.audio_mixer:
                        await self.audio_mixer.play(procedural_clip)

                    self.logger.info(
                        f"Procedural animation {procedural_type.name} started playing."
                    )

                # Get current animation frame
                assert procedural_clip.start_time is not None
                animation_time = current_time - procedural_clip.start_time
                animation_frame = procedural_clip.get_frame(animation_time)
                if animation_frame is None:
                    continue  # Ignore if there is no frame

                # Initialize procedural frame
                procedural_frame: Frame | None = None

                # Procedural animation might be finishing
                if (
                    procedural_clip.end_time is not None
                    and not procedural_clip.is_paused()
                ):
                    # Stop procedural animation
                    if current_time > procedural_clip.end_time:
                        stop_procedural.append(procedural_type)

                        # Save the body rotation
                        if procedural_type == ProceduralType.LOOK_AT:
                            assert animation_frame.body_angle is not None
                            self.static_body_angle = animation_frame.body_angle
                        continue

                    # Procedural animation is blending out
                    procedural_frame = procedural_clip.blend_out(animation_time)

                # Procedural animation is blending in
                if procedural_frame is None:
                    procedural_frame = procedural_clip.blend_in(animation_time)

                # Add the procedural frame to the composed frame
                frame_to_add = (
                    animation_frame if procedural_frame is None else procedural_frame
                )
                frame_to_add.clamp_frame(self.config.joint_limits)
                new_composed_frame = procedural_clip.add_to_frame(
                    frame_to_add, new_composed_frame
                )

            # Remove procedural animations
            for procedural_type in stop_procedural:
                # Mark it as finished
                await self.send_clip_finished(
                    self.procedural_clips[procedural_type].action_uuid,
                    current_time,
                )

                # Remove it from the list
                self.procedural_clips.pop(procedural_type)
                self.logger.info(
                    f"Procedural animation {procedural_type.name} finished."
                )

            # Add the static body angle frame to the composed frame
            if self.static_body_angle is not None:
                new_composed_frame = new_composed_frame.additive_blend(
                    Frame(body_angle=self.static_body_angle)
                )

        return new_composed_frame

    #############################
    # Clip status
    ##############################

    async def send_clip_finished(self, action_uuid: str, time: float) -> None:
        await self.publish(
            clip_status_topic,
            ClipStatus(
                action_uuid=action_uuid,
                robot_id=self.robot_id.to_proto(),
                status=ClipStatus.FINISHED,
                timestamp=int(time * 1000),
            ),
        )

    async def send_clip_started(self, action_uuid: str, time: float) -> None:
        await self.publish(
            clip_status_topic,
            ClipStatus(
                action_uuid=action_uuid,
                robot_id=self.robot_id.to_proto(),
                status=ClipStatus.STARTED,
                timestamp=int(time * 1000),
            ),
        )

    async def send_clip_updated(self, action_uuid: str, time: float) -> None:
        await self.publish(
            clip_status_topic,
            ClipStatus(
                action_uuid=action_uuid,
                robot_id=self.robot_id.to_proto(),
                status=ClipStatus.UPDATED,
                timestamp=int(time * 1000),
            ),
        )

    async def send_clip_error(self, action_uuid: str) -> None:
        await self.publish(
            clip_status_topic,
            ClipStatus(
                action_uuid=action_uuid,
                robot_id=self.robot_id.to_proto(),
                status=ClipStatus.ERROR,
                timestamp=int(time.time() * 1000),
            ),
        )


########################
# Main
########################


async def main() -> None:
    config = load_config(CompositorConfig)
    await ServiceExecutor([AnimationCompositorService(config)]).run()


if __name__ == "__main__":
    asyncio.run(main())
