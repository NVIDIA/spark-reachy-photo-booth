# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import override

import pydub
from clips.animations import Animation, EulerAngles, Frame
from clips.base_clips import AnimationClip, AudioClip, BaseClip
from workmesh.messages import AnimationFrame as AnimationFrameProto
from workmesh.messages import Audio
from workmesh.messages import ProceduralType as ProceduralTypeProto


# NOTE: we can add more procedural animations here
class ProceduralType(Enum):
    LOOK_AT = "look_at"
    TRACK = "track"

    @classmethod
    def from_proto(
        cls,
        procedural_type: ProceduralTypeProto,
    ) -> "ProceduralType":
        if procedural_type == ProceduralTypeProto.LOOK_AT:
            return ProceduralType.LOOK_AT
        elif procedural_type == ProceduralTypeProto.TRACK:
            return ProceduralType.TRACK
        else:
            raise ValueError(f"Unknown procedural type: {procedural_type}")

    @classmethod
    def to_proto(cls, procedural_type: "ProceduralType") -> ProceduralTypeProto:
        try:
            return getattr(ProceduralTypeProto, procedural_type.name)
        except AttributeError as e:
            raise ValueError(f"Unknown procedural type: {procedural_type.name}") from e


class ProceduralAnimation(AnimationClip):
    """Procedural animation."""

    def __init__(
        self,
        animation_data: Animation | None = None,
        blend_in_duration: float = 0.0,
        blend_out_duration: float = 0.0,
    ):
        super().__init__(
            animation_data=animation_data,
            blend_in_duration=blend_in_duration,
            blend_out_duration=blend_out_duration,
        )


class ProceduralClip(BaseClip, ABC):
    """Procedural animation."""

    def __init__(self, action_uuid: str, blend_in_duration: float = 0.0):
        self.action_uuid: str = action_uuid
        self.animation = ProceduralAnimation(blend_in_duration=blend_in_duration)

        # Paused state
        self._paused: bool = False
        self._paused_time: float | None = None
        self._blend_in_frame: Frame | None = None

    def start(self, start_time: float):
        """Defines the exact start of the animation by setting the start_time."""
        self.start_time = start_time

    def stop(self, blend_out_duration: float):
        """Stop the animation.

        Arguments:
            blend_out_duration (float): The duration of the blend out.
        """

        # Ignore if the animation hasn't started
        if self.start_time is None:
            return

        self._paused = False

        assert self.animation is not None
        self.animation.blend_out_duration = blend_out_duration
        self.end_time = time.time() + blend_out_duration
        self.animation.duration = self.end_time - self.start_time

    def pause(self, enable: bool):
        """Pause the animation.

        Arguments:
            enable (bool): True if we pause the animation, false otherwise.
        """

        if enable:
            assert self.start_time is not None
            self._paused_time = time.time()
            self._blend_in_frame = self.get_frame(self._paused_time - self.start_time)

        self._paused = enable

    def is_paused(self) -> bool:
        """Check if the animation is paused.

        Returns:
            bool: True if the animation is paused, otherwise False.
        """
        return self._paused

    def is_blending_in(self, time: float) -> bool:
        """Check if the procedural animation is blending in.

        Arguments:
            time (float): The current time (seconds) since the start of the animation.

        Returns:
            bool: True if the animation is blending in, otherwise False.
        """

        return self.animation is not None and self.animation.is_blending_in(time)

    def is_blending_out(self, time: float) -> bool:
        """Check if the procedural animation is blending out.

        Arguments:
            time (float): The current time (seconds) since the start of the animation.

        Returns:
            bool: True if the animation is blending out, otherwise False.
        """

        return self.animation is not None and self.animation.is_blending_out(time)

    def blend_in(self, time: float) -> Frame | None:
        """Blend in the procedural animation.

        If the animation is currently blending in, returns a frame that is
        interpolated between the animation frame and the reference pose,
        weighted by the blend-in progress. Otherwise, returns None.

        Arguments:
            time (float): The current time (seconds) since the start of the animation.

        Returns:
            Frame | None: The blended frame if blending in, otherwise None.
        """

        if not self.is_blending_in(time) or self.is_paused():
            return None

        assert self.animation is not None
        weight = 1 - time / self.animation.blend_in_duration

        animation_frame = self.get_frame(time)
        assert animation_frame is not None

        # If the animation was paused before, blend in with previous pose
        frame_to_blend = self._blend_in_frame
        if frame_to_blend is None:
            frame_to_blend = Frame.reference_pose()

        return animation_frame.blend_to_frame(frame_to_blend, weight)

    def blend_out(self, time: float) -> Frame | None:
        """Blend out the procedural animation.

        If the animation is currently blending out, returns a frame that is
        interpolated between the animation frame and the reference pose,
        weighted by the blend-out progress. Otherwise, returns None.

        Arguments:
            time (float): The current time (seconds) since the start of the animation.

        Returns:
            Frame | None: The blended frame if blending out, otherwise None.
        """

        if not self.is_blending_out(time) or self.is_paused():
            return None

        assert self.animation is not None
        assert self.animation.duration is not None
        blend_out_elapsed = self.animation.duration - time
        weight = 1 - blend_out_elapsed / self.animation.blend_out_duration

        animation_frame = self.get_frame(blend_out_elapsed)
        assert animation_frame is not None
        return animation_frame.blend_to_frame(Frame.reference_pose(), weight)

    @abstractmethod
    def update_animation(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_frame(self, time: float) -> Frame | None:
        """Get the frame at the given time.

        Arguments:
            time (float): The current time (seconds) since the start of the animation.

        Returns:
            Frame | None: The frame at the given time. If not available, returns None.
        """

        raise NotImplementedError

    def add_frame(self, frame: Frame) -> None:
        """Add a frame to the animation data.

        Arguments:
            frame (Frame): The frame to add.
        """

        assert self.animation is not None
        if self.animation.animation_data is None:
            self.animation.animation_data = Animation(frames=[frame])
        else:
            self.animation.animation_data.frames.append(frame)

    def get_frame_by_index(self, index: int) -> Frame | None:
        """Get a frame by index from the animation data.

        Args:
            index (int): The index of the frame to retrieve.

        Returns:
            Frame | None: The frame at the specified index, or None if the index
                is out of bounds or the animation data is missing.
        """
        assert self.animation is not None
        if self.animation.animation_data is None:
            return None

        if index < 0:
            index = len(self.animation.animation_data.frames) + index

        # Index is out of bounds
        if not (0 <= index < len(self.animation.animation_data.frames)):
            raise ValueError(
                f"Index {index} is out of bounds for animation with "
                f"{len(self.animation.animation_data.frames)} frames"
            )

        return self.animation.animation_data.frames[index]

    @abstractmethod
    def add_to_frame(self, frame: Frame, composed_frame: Frame) -> Frame:
        """Add the procedural frame to the composed frame.

        Arguments:
            frame (Frame): The procedural frame.
            composed_frame (Frame): The composed frame.
        """

        raise NotImplementedError

    def set_blend_in_frame(self, frame: Frame | None) -> None:
        """Set the blend in frame.

        Arguments:
            frame (Frame | None): The blend in frame.
        """
        self._blend_in_frame = frame


class LookAtAnimation(ProceduralClip):
    # NOTE: This animation doesn't do blend in or out
    # since it is already a movement to a target position
    # It also doesn't stop manually, it will stop when the target is reached

    def __init__(self, action_uuid: str):
        super().__init__(action_uuid)
        self.duration: float | None = None

    @override
    def start(self, start_time: float) -> None:
        if self.animation is None or self.duration is None:
            return

        super().start(start_time)
        self.end_time = start_time + self.duration

    def update_animation(
        self,
        animation: list[AnimationFrameProto],
        audio: Audio | None,
        duration: float,
        volume: float | None = None,
    ):
        """Update the animation with the new frames.

        Arguments:
            frames (list[AnimationFrameProto]): The list of animation frames.
            duration (float): The duration of the animation.
        """
        assert self.animation is not None

        self.animation.animation_data = Animation(
            frames=[Frame(body_angle=frame.body_angle) for frame in animation],
        )
        self.animation.duration = duration
        self.duration = duration

        if audio is not None:
            self.audio = AudioClip(
                pydub.AudioSegment(
                    audio.audio_buffer,
                    sample_width=audio.bits_per_sample // 8,
                    frame_rate=audio.sample_rate,
                    channels=audio.channel_count,
                )
            )
            if volume is not None:
                self.audio.change_volume(volume)

    def get_frame(self, time: float) -> Frame | None:
        assert self.animation is not None
        if self.duration is None or self.animation.animation_data is None:
            return None

        if self.is_paused():
            return self._blend_in_frame

        return self.animation.animation_data.eval_frame(self.duration, time)

    @override
    def pause(self, enable: bool) -> None:
        super().pause(enable)

        if not enable and self.duration is not None:
            assert self._paused_time is not None
            assert self.start_time is not None
            assert self.end_time is not None

            time_paused = time.time() - self._paused_time
            self.end_time += time_paused
            self.start_time += time_paused

            self._paused_time = None
        else:
            self._paused_time = time.time()

    def add_to_frame(self, frame: Frame, composed_frame: Frame) -> Frame:
        """Add the procedural frame to the composed frame.
        In the case of LOOK_AT, it will replace the body angle.

        Arguments:
            frame (Frame): The procedural frame.
            composed_frame (Frame): The composed frame.
        """

        composed_frame.body_angle = frame.body_angle
        return composed_frame


class TrackAnimation(ProceduralClip):
    def __init__(self, action_uuid: str, blend_in_duration: float = 0.0):
        super().__init__(action_uuid, blend_in_duration)

    def update_animation(self, frames: list[AnimationFrameProto]):
        """Update the animation with the new frames.

        Arguments:
            frames (list[AnimationFrameProto]): The list of animation frames.
        """

        if len(frames) == 0:
            return

        # NOTE: Uses only the first frame because this procedural animation
        # is created frame by frame so it only contains one frame
        frame = Frame(
            head_rotation=EulerAngles(
                roll=frames[0].head_rotation.roll,
                pitch=frames[0].head_rotation.pitch,
                yaw=frames[0].head_rotation.yaw,
            ),
            body_angle=frames[0].body_angle,
        )

        if self._blend_in_frame is not None:
            frame = self._blend_in_frame.additive_blend(frame)

        assert self.animation is not None
        self.animation.animation_data = Animation(
            frames=[frame],
        )

    @override
    def pause(self, enable: bool) -> None:
        super().pause(enable)

        if not enable:
            self.start_time = time.time()

    def get_frame(self, time: float) -> Frame | None:
        assert self.animation is not None
        if self.is_paused():
            return self._blend_in_frame

        return (
            None
            if self.animation.animation_data is None
            else self.animation.animation_data.frames[0]
        )

    def add_to_frame(self, frame: Frame, composed_frame: Frame) -> Frame:
        """Add the procedural frame to the composed frame.
        In the case of TRACK, it will only blend additively
        with the current frame since the TRACK frame
        is an increment of the previous frame.

        Arguments:
            frame (Frame): The procedural frame.
            composed_frame (Frame): The composed frame.
        """

        return composed_frame.additive_blend(frame)
