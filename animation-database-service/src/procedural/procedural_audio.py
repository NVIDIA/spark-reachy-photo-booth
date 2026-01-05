# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Procedural audio generation for robot animations.

This module provides the main API for generating robot sounds from velocity data or
based on duration.
"""

import easing_curves
import numpy as np
from audio import audio_engine
from configuration import DEFAULT_SAMPLE_RATE


def get_duration(
    duration: float | None = None,
    speed_normalized: np.ndarray | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> float:
    """Calculate duration from either explicit duration or speed_normalized array.

    Args:
        duration: Duration in seconds (optional)
        speed_normalized: Velocity envelope array (optional)
        sample_rate: Sample rate in Hz (default: from configuration module)

    Returns:
        Duration in seconds

    Raises:
        ValueError: If neither duration nor speed_normalized is provided
    """
    if duration is None and speed_normalized is None:
        raise ValueError("Either duration or speed_normalized must be provided")

    if duration is None:
        assert speed_normalized is not None, (
            "speed_normalized must be provided if duration is None"
        )
        return speed_normalized.shape[0] / sample_rate

    return duration


def generate_body_angle_sound(
    duration: float | None = None,
    speed_normalized: np.ndarray | None = None,
    volume: float = 1.0,
) -> np.ndarray:
    """Generate sound for base rotation (yaw axis).

    Args:
        duration: Duration in seconds (if None, calculated from speed_normalized)
        speed_normalized: Velocity envelope 0-1 (if None, uses flat envelope)
        volume: Volume multiplier

    Returns:
        Audio as numpy array
    """
    duration = get_duration(duration, speed_normalized)

    # Descending pitch arc - falling then rising
    gen = audio_engine.FrequencyGenerator(
        base_freq=240 + np.random.uniform(-10, 10),
        duration=duration,
        volume=volume * 0.04,
        volume_fade_in_duration=0.38,
        volume_fade_out_duration=0.80,
        volume_fade_in_interp_func=easing_curves.ease_out_elastic,
        volume_fade_out_interp_func=easing_curves.ease_in_quad,
        volume_envelope=speed_normalized,
        pitch_fade_in_duration=0.25,
        pitch_fade_out_duration=0.25,
        pitch_fade_in_interp_func=easing_curves.ease_in_cubic,
        pitch_fade_out_interp_func=easing_curves.ease_out_cubic,
        pitch_fade_range=(0.60, 1.10),
        pitch_envelope=speed_normalized,
    )
    sound = gen.apply_effects(
        audio_engine.HarmonicsEffect([(2, 0.18), (3, 0.12), (5, 0.06)]),
        audio_engine.TremoloEffect(rate=2.0, depth=0.8),
        audio_engine.SmoothingEffect(smoothing_time=0.008),
    )

    gen = audio_engine.FrequencyGenerator(
        base_freq=860 + np.random.uniform(-50, 50),
        duration=duration,
        volume=volume * 0.04,
        volume_fade_in_duration=0.35,
        volume_fade_out_duration=0.70,
        volume_envelope=speed_normalized,
        pitch_fade_range=(0.80, 1.20),
        pitch_envelope=speed_normalized,
    )
    sound += gen.apply_effects(
        audio_engine.PulseEffect(pulse_rate=32.0, duty_cycle=0.28, fade_time=0.006),
        audio_engine.BitCrushEffect(bit_depth=5),
        audio_engine.SmoothingEffect(smoothing_time=0.003),
    )

    gen = audio_engine.FrequencyGenerator(
        base_freq=3100 + np.random.uniform(-56, 56),
        duration=duration,
        volume=volume * 0.075,
        volume_envelope=speed_normalized,
        pitch_fade_range=(0.85, 1.05),
        pitch_envelope=speed_normalized,
    )
    sound += gen.apply_effects(
        audio_engine.PulseEffect(pulse_rate=12.8, duty_cycle=0.028, fade_time=0.0009),
        audio_engine.NoiseEffect(amount=0.52, high_pass_freq=1950),
        audio_engine.SmoothingEffect(smoothing_time=0.0032),
    )

    return sound


def generate_head_yaw_sound(
    duration: float | None = None,
    speed_normalized: np.ndarray | None = None,
    volume: float = 1.0,
) -> np.ndarray:
    """Generate sound for head yaw (left/right rotation).

    Args:
        duration: Duration in seconds (if None, calculated from speed_normalized)
        speed_normalized: Velocity envelope 0-1 (if None, uses flat envelope)
        volume: Volume multiplier

    Returns:
        Audio as numpy array
    """
    duration = get_duration(duration, speed_normalized)

    gen = audio_engine.FrequencyGenerator(
        base_freq=650 + np.random.uniform(-20, 20),
        duration=duration,
        volume=volume * 1.25,
        volume_fade_in_duration=0.5,
        volume_fade_out_duration=1.5,
        volume_fade_in_interp_func=easing_curves.ease_out_cubic,
        volume_fade_out_interp_func=easing_curves.ease_out_cubic,
        volume_envelope=speed_normalized,
        pitch_fade_in_duration=1.4,
        pitch_fade_out_duration=1.2,
        pitch_fade_in_interp_func=easing_curves.ease_out_cubic,
        pitch_fade_out_interp_func=easing_curves.ease_out_cubic,
        pitch_fade_range=(0.7, 1.0),
        pitch_envelope=speed_normalized,
    )
    sound = gen.apply_effects(
        audio_engine.HarmonicsEffect([(2, 0.2), (3, 0.1), (4, 0.05)]),
        audio_engine.SmoothingEffect(smoothing_time=0.002),
        audio_engine.TremoloEffect(rate=15.0, depth=2),
        audio_engine.WahWahEffect(min_freq=1000, max_freq=1500),
        audio_engine.PulseEffect(pulse_rate=30.0, duty_cycle=0.7, fade_time=0.02),
        audio_engine.BitCrushEffect(bit_depth=6),
        audio_engine.SmoothingEffect(smoothing_time=0.002),
    )

    return sound


def generate_head_pitch_sound(
    duration: float | None = None,
    speed_normalized: np.ndarray | None = None,
    volume: float = 1.0,
) -> np.ndarray:
    """Generate sound for head pitch (up/down tilt).

    Args:
        duration: Duration in seconds (if None, calculated from speed_normalized)
        speed_normalized: Velocity envelope 0-1 (if None, uses flat envelope)
        volume: Volume multiplier

    Returns:
        Audio as numpy array
    """
    duration = get_duration(duration, speed_normalized)

    # Primary layer
    gen = audio_engine.FrequencyGenerator(
        base_freq=520 + np.random.uniform(-24, 24),
        duration=duration,
        volume=volume * 1,
        volume_fade_in_duration=0.5,
        volume_fade_out_duration=1.3,
        volume_fade_in_interp_func=easing_curves.ease_out_cubic,
        volume_fade_out_interp_func=easing_curves.ease_out_cubic,
        volume_envelope=speed_normalized,
        pitch_fade_in_duration=1.1,
        pitch_fade_out_duration=1.0,
        pitch_fade_in_interp_func=easing_curves.ease_out_quad,
        pitch_fade_out_interp_func=easing_curves.ease_out_cubic,
        pitch_fade_range=(0.72, 1.02),
        pitch_envelope=speed_normalized,
    )
    sound = gen.apply_effects(
        audio_engine.HarmonicsEffect([(2, 0.22), (3, 0.14), (5, 0.05)]),
        audio_engine.PulseEffect(pulse_rate=28.0, duty_cycle=0.68, fade_time=0.022),
        audio_engine.TremoloEffect(rate=16.0, depth=2.2),
        audio_engine.SmoothingEffect(smoothing_time=0.02),
    )

    return sound


def generate_head_roll_sound(
    duration: float | None = None,
    speed_normalized: np.ndarray | None = None,
    volume: float = 1.0,
) -> np.ndarray:
    """Generate sound for head roll (ear-to-shoulder tilt).

    Args:
        duration: Duration in seconds (if None, calculated from speed_normalized)
        speed_normalized: Velocity envelope 0-1 (if None, uses flat envelope)
        volume: Volume multiplier

    Returns:
        Audio as numpy array
    """
    duration = get_duration(duration, speed_normalized)

    gen = audio_engine.FrequencyGenerator(
        base_freq=600 + np.random.uniform(-20, 20),
        duration=duration,
        volume=volume * 1,
        volume_fade_in_duration=0.03,
        volume_fade_out_duration=0.12,
        volume_fade_in_interp_func=easing_curves.ease_out_back,
        volume_fade_out_interp_func=easing_curves.ease_in_cubic,
        volume_envelope=speed_normalized,
        pitch_fade_in_duration=0.06,
        pitch_fade_out_duration=0.18,
        pitch_fade_in_interp_func=easing_curves.ease_out_elastic,
        pitch_fade_out_interp_func=easing_curves.ease_in_quad,
        pitch_fade_range=(1.5, 0.85),
        pitch_envelope=speed_normalized,
    )
    sound = gen.apply_effects(
        audio_engine.HarmonicsEffect([(2, 0.2), (3, 0.12)]),
        audio_engine.PulseEffect(pulse_rate=15.0, duty_cycle=0.4, fade_time=0.04),
        audio_engine.PitchBendEffect(bend_amount=1.4, bend_rate=12.0),
        audio_engine.SmoothingEffect(smoothing_time=0.025),
    )

    return sound


def generate_l_antenna_angle_sound(
    duration: float | None = None,
    speed_normalized: np.ndarray | None = None,
    volume: float = 1.0,
) -> np.ndarray:
    """Generate sound for left antenna

    Args:
        duration: Duration in seconds (if None, calculated from speed_normalized)
        speed_normalized: Velocity envelope 0-1 (if None, uses flat envelope)
        volume: Volume multiplier
        variant: Sound variant to generate (1-12)

    Returns:
        Audio as numpy array
    """
    duration = get_duration(duration, speed_normalized)

    # SOFTER - More gentle, extended fade
    gen = audio_engine.FrequencyGenerator(
        base_freq=2000 + np.random.uniform(-125, 125),
        duration=duration,
        volume=volume * 0.35,
        volume_envelope=speed_normalized,
        pitch_fade_range=(0.83, 1.27),
        pitch_envelope=speed_normalized,
    )
    sound = gen.apply_effects(
        audio_engine.HarmonicsEffect([(3, 0.08), (7, 0.04)]),
        audio_engine.PulseEffect(pulse_rate=1.1, duty_cycle=0.10, fade_time=0.018),
        audio_engine.BitCrushEffect(bit_depth=5),
        audio_engine.SmoothingEffect(smoothing_time=0.085),
    )
    return sound


def generate_r_antenna_angle_sound(
    duration: float | None = None,
    speed_normalized: np.ndarray | None = None,
    volume: float = 1.0,
) -> np.ndarray:
    """Generate sound for right antenna

    Args:
        duration: Duration in seconds (if None, calculated from speed_normalized)
        speed_normalized: Velocity envelope 0-1 (if None, uses flat envelope)
        volume: Volume multiplier
        variant: Sound variant to generate (1-12)

    Returns:
        Audio as numpy array
    """
    duration = get_duration(duration, speed_normalized)

    # SOFTER - Gentler bubbles
    gen = audio_engine.FrequencyGenerator(
        base_freq=250 + np.random.uniform(-46, 46),
        duration=duration,
        volume=volume * 0.85,
        volume_envelope=speed_normalized,
        pitch_fade_range=(0.53, 1.48),
        pitch_envelope=speed_normalized,
    )
    sound = gen.apply_effects(
        audio_engine.HarmonicsEffect([(5, 0.20), (7, 0.12), (11, 0.06)]),
        audio_engine.PulseEffect(pulse_rate=0.8, duty_cycle=0.16, fade_time=0.014),
        audio_engine.NoiseEffect(amount=0.35, high_pass_freq=1150),
        audio_engine.SmoothingEffect(smoothing_time=0.072),
    )
    return sound
