# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pydub
from clips.animations import Animation, Frame

########################
# Base clips
########################


@dataclass()
class AudioClip:
    """Audio clip."""

    audio_data: pydub.AudioSegment

    volume: float = 1.0
    fade_in_duration: float = 0.0  # in ms
    fade_out_duration: float = 0.0  # in ms

    n_bytes_read: int = 0

    def __post_init__(self) -> None:
        self.initial_volume_dB = self.audio_data.dBFS
        self.original_audio_data = deepcopy(self.audio_data)

    def change_volume(self, volume: float) -> None:
        self.volume = volume
        self.audio_data = self.original_audio_data + 20 * np.log10(volume)

    def is_fading_out(self, t: float) -> bool:
        """Check if the audio is fading out."""
        return (
            self.fade_out_duration > 0
            and t > self.audio_data.duration_seconds - self.fade_out_duration
        )

    def is_fading_in(self, t: float) -> bool:
        """Check if the audio is fading in."""
        return (
            self.fade_in_duration > 0
            and t < self.audio_data.duration_seconds - self.fade_in_duration
        )


@dataclass()
class AnimationClip:
    """Animation clip."""

    animation_data: Animation | None = None

    duration: float | None = None

    blend_in_duration: float = 0.5  # in seconds
    blend_out_duration: float = 0.5  # in seconds

    def is_blending_in(self, time: float) -> bool:
        """Check if the animation is blending in.

        Arguments:
            time (float): The current time (seconds) since the start of the animation.

        Returns:
            bool: True if the animation is blending in, otherwise False.
        """

        return self.blend_in_duration > 0.0 and time <= self.blend_in_duration

    def is_blending_out(self, time: float) -> bool:
        """Check if the animation is blending out.

        Arguments:
            time (float): The current time (seconds) since the start of the animation.

        Returns:
            bool: True if the animation is blending out, otherwise False.
        """

        return (
            self.duration is not None
            and self.blend_out_duration > 0.0
            and self.duration - self.blend_out_duration < time <= self.duration
        )


@dataclass()
class BaseClip(ABC):
    """Clip item."""

    action_uuid: str

    audio: AudioClip | None = None
    animation: AnimationClip | None = None

    start_time: float | None = None
    end_time: float | None = None

    def in_transition_in(self, time: float) -> bool:
        """
        Return True if the clip is currently in its transition-in (blend-in or fade-in)
            phase at the given time.

        Args:
            time (float): The current time (in seconds) since the start of the clip.
        """

        return (self.animation is not None and self.animation.is_blending_in(time)) or (
            self.audio is not None and self.audio.is_fading_in(time)
        )

    def in_transition_out(self, time: float) -> bool:
        """
        Return True if the clip is currently in its transition-out (blend-out or fade-out)
        phase at the given time.

        Args:
            time (float): The current time (in seconds) since the start of the clip.
        """  # noqa: E501

        return (
            self.animation is not None and self.animation.is_blending_out(time)
        ) or (self.audio is not None and self.audio.is_fading_out(time))


########################
# Playing clips
########################


@dataclass(order=True)
class BlendingAnimationClip:
    """Blending animation clip."""

    priority: int

    frame: Frame = field(compare=False)
    weight: np.ndarray = field(compare=False)
