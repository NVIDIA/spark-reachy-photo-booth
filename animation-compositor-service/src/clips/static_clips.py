# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from clips.animations import Animation
from clips.base_clips import AnimationClip, AudioClip, BaseClip


class StaticAnimation(AnimationClip):
    """Animation clip."""

    def __init__(
        self,
        animation_data: Animation,
        frame_rate: int,
        duration: float,
        blend_in_duration: float = 0.5,
        blend_out_duration: float = 0.5,
        priority: int = 0,
        opacity: float = 1.0,
    ):
        super().__init__(
            animation_data=animation_data,
            duration=duration,
            blend_in_duration=blend_in_duration,
            blend_out_duration=blend_out_duration,
        )
        self.frame_rate = frame_rate

        self.priority = priority
        self.opacity = opacity

        self.finished_blend_in: bool = False

    def cut(self, t: float) -> None:
        """Cut the animation at a specific time."""

        assert self.animation_data is not None
        self.animation_data.frames = self.animation_data.frames[
            : int(t * self.frame_rate) + 1
        ]
        self.duration = t

    def is_blending_in(self, time: float) -> bool:
        """Check if the animation is blending in."""
        return not self.finished_blend_in and super().is_blending_in(time)


class StaticClip(BaseClip):
    """Static clip."""

    def __init__(
        self,
        action_uuid: str,
        duration: float,
        animation: StaticAnimation | None = None,
        audio: AudioClip | None = None,
        loop: bool = False,
        loop_overlap: float = 1.0,
    ):
        super().__init__(action_uuid=action_uuid, animation=animation, audio=audio)

        # Loop properties
        self.loop: bool = loop
        self.loop_overlap: float = loop_overlap

        # Total clip duration: max of animation and audio duration
        self.duration: float = duration
