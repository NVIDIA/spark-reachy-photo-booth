# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core audio generation engine with frequency generators and effects.

This module provides the foundation for procedural audio generation:
- FrequencyGenerator: Generate sine waves with pitch and volume envelopes
- AudioEffect: Base class for audio processing effects
- Built-in effects: Wobble, Harmonics, Distortion, Echo, WahWah, Tremolo, Pulse,
  Noise, PitchBend, BitCrush, Stutter, Smoothing
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import easing_curves
import numpy as np
from configuration import DEFAULT_SAMPLE_RATE


def create_envelope(
    t: np.ndarray,
    duration: float,
    sample_rate: int,
    fade_in_duration: float,
    fade_out_duration: float,
    fade_in_interp_func: Callable[[float], float],
    fade_out_interp_func: Callable[[float], float] | None = None,
) -> np.ndarray:
    """Create fade in/out envelope using easing curves.

    Args:
        t: Time array
        duration: Total duration in seconds
        sample_rate: Sample rate in Hz
        fade_in_duration: Fade in duration in seconds
        fade_out_duration: Fade out duration in seconds
        fade_in_interp_func: Easing function for fade in
        fade_out_interp_func: Easing function for fade out (uses fade_in if None)

    Returns:
        Envelope array with fade in/out applied
    """
    envelope = np.ones_like(t)

    if fade_out_interp_func is None:
        fade_out_interp_func = fade_in_interp_func

    fade_in_samples = int(min(fade_in_duration, duration / 2) * sample_rate)
    fade_out_samples = int(min(fade_out_duration, duration / 2) * sample_rate)

    if fade_in_samples + fade_out_samples > len(t):
        total_samples = len(t)
        fade_in_samples = total_samples // 2
        fade_out_samples = total_samples - fade_in_samples

    if fade_in_samples > 0:
        progress = np.linspace(0, 1, fade_in_samples)
        fade_in_curve = [fade_in_interp_func(p) for p in progress]
        envelope[:fade_in_samples] = np.array(fade_in_curve)

    if fade_out_samples > 0:
        progress = np.linspace(0, 1, fade_out_samples)
        fade_out_curve = [fade_out_interp_func(1 - p) for p in progress]
        envelope[-fade_out_samples:] = np.array(fade_out_curve)

    return envelope


class FrequencyGenerator:
    """Generate sine waves with controllable pitch and volume envelopes.

    Supports both fade-based envelopes (using easing curves) and custom
    velocity-based envelopes for animation-driven audio.
    """

    def __init__(
        self,
        base_freq: float,
        duration: float,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        volume: float = 1.0,
        volume_fade_in_duration: float = 0.0,
        volume_fade_out_duration: float = 0.0,
        volume_fade_in_interp_func: Callable[[float], float] = easing_curves.linear,
        volume_fade_out_interp_func: Callable[[float], float] | None = None,
        volume_envelope: np.ndarray | None = None,
        pitch_fade_in_duration: float = 0.0,
        pitch_fade_out_duration: float = 0.0,
        pitch_fade_in_interp_func: Callable[[float], float] = easing_curves.linear,
        pitch_fade_out_interp_func: Callable[[float], float] | None = None,
        pitch_fade_range: tuple[float, float] = (0.7, 1.0),
        pitch_envelope: np.ndarray | None = None,
    ) -> None:
        """Initialize frequency generator.

        Args:
            base_freq: Base frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz (default: from configuration module)
            volume: Base volume 0.0-1.0 (default: 1.0)
            volume_fade_in_duration: Volume fade in time in seconds
            volume_fade_out_duration: Volume fade out time in seconds
            volume_fade_in_interp_func: Easing function for volume fade in
            volume_fade_out_interp_func: Easing function for fade out
                (uses fade_in if None)
            volume_envelope: Custom volume envelope 0-1, overrides fades
                if provided
            pitch_fade_in_duration: Pitch fade in time in seconds
            pitch_fade_out_duration: Pitch fade out time in seconds
            pitch_fade_in_interp_func: Easing function for pitch fade in
            pitch_fade_out_interp_func: Easing function for fade out
                (uses fade_in if None)
            pitch_fade_range: (min, max) pitch multipliers, e.g. (0.7, 1.0)
            pitch_envelope: Custom pitch envelope 0-1, overrides fades
                if provided
        """
        self.base_freq = base_freq
        self.duration = duration
        self.sample_rate = sample_rate
        self.volume = volume
        self.volume_fade_in_duration = volume_fade_in_duration
        self.volume_fade_out_duration = volume_fade_out_duration
        self.volume_fade_in_interp_func = volume_fade_in_interp_func
        self.volume_fade_out_interp_func = volume_fade_out_interp_func
        self.pitch_fade_in_duration = pitch_fade_in_duration
        self.pitch_fade_out_duration = pitch_fade_out_duration
        self.pitch_fade_in_interp_func = pitch_fade_in_interp_func
        self.pitch_fade_out_interp_func = pitch_fade_out_interp_func
        self.pitch_fade_range = pitch_fade_range

        self.t = np.linspace(0, duration, int(sample_rate * duration), False)
        self.base_frequency = base_freq + np.zeros_like(self.t)
        self.frequency = self.base_frequency.copy()
        self._custom_volume_envelope = volume_envelope
        self._custom_pitch_envelope = pitch_envelope

        if pitch_envelope is not None:
            pitch_env = self._resample_envelope(pitch_envelope, len(self.t))
            pitch_min, pitch_max = pitch_fade_range
            pitch_multiplier = pitch_min + (pitch_max - pitch_min) * pitch_env
            self.frequency = self.frequency * pitch_multiplier
        elif pitch_fade_in_duration > 0 or pitch_fade_out_duration > 0:
            pitch_env = create_envelope(
                self.t,
                duration,
                sample_rate,
                pitch_fade_in_duration,
                pitch_fade_out_duration,
                pitch_fade_in_interp_func,
                pitch_fade_out_interp_func,
            )
            pitch_min, pitch_max = pitch_fade_range
            pitch_multiplier = pitch_min + (pitch_max - pitch_min) * pitch_env
            self.frequency = self.frequency * pitch_multiplier

    def _resample_envelope(
        self, envelope: np.ndarray, target_length: int
    ) -> np.ndarray:
        """Resample envelope to target length using linear interpolation."""
        if len(envelope) == target_length:
            return envelope

        old_indices = np.linspace(0, len(envelope) - 1, len(envelope))
        new_indices = np.linspace(0, len(envelope) - 1, target_length)
        return np.interp(new_indices, old_indices, envelope)

    def generate(self) -> np.ndarray:
        """Generate base audio signal with envelopes applied.

        Returns:
            Audio signal as numpy array
        """
        phase = np.cumsum(2 * np.pi * self.frequency / self.sample_rate)
        sound = np.sin(phase) * self.volume

        if self._custom_volume_envelope is not None:
            volume_env = self._resample_envelope(
                self._custom_volume_envelope, len(self.t)
            )
            sound *= volume_env
        elif self.volume_fade_in_duration > 0 or self.volume_fade_out_duration > 0:
            volume_env = create_envelope(
                self.t,
                self.duration,
                self.sample_rate,
                self.volume_fade_in_duration,
                self.volume_fade_out_duration,
                self.volume_fade_in_interp_func,
                self.volume_fade_out_interp_func,
            )
            sound *= volume_env

        return sound

    def apply_effects(self, *effects: "AudioEffect") -> np.ndarray:
        """Generate audio and apply a chain of effects.

        Args:
            *effects: Variable number of AudioEffect instances to apply

        Returns:
            Processed audio signal as numpy array
        """
        sound = self.generate()
        for effect in effects:
            sound = effect.apply(sound, self.t, self.frequency, self.sample_rate)
        return sound


class AudioEffect(ABC):
    """Base class for audio effects."""

    @abstractmethod
    def apply(
        self, audio: np.ndarray, t: np.ndarray, frequency: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Apply the effect to the audio signal.

        Args:
            audio: Input audio signal
            t: Time array
            frequency: Frequency array for frequency-dependent effects
            sample_rate: Sample rate in Hz

        Returns:
            Modified audio signal
        """
        pass


class WobbleEffect(AudioEffect):
    """Add frequency wobble/vibrato for character."""

    def __init__(self, wobbles: list[tuple[float, float]], mix: float = 0.3) -> None:
        """Initialize wobble effect.

        Args:
            wobbles: List of (amplitude, frequency) tuples, e.g. [(5, 0.8), (3, 1.5)]
            mix: Mix amount 0.0-1.0 (default: 0.3)
        """
        self.wobbles = wobbles
        self.mix = mix

    def apply(
        self, audio: np.ndarray, t: np.ndarray, frequency: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Apply wobble by adding modulated sine waves."""
        result = audio.copy()

        for amplitude, wobble_freq in self.wobbles:
            wobbled_freq = frequency + amplitude * np.sin(2 * np.pi * wobble_freq * t)
            phase = np.cumsum(2 * np.pi * wobbled_freq / sample_rate)
            wobble_component = np.sin(phase)
            result += wobble_component * self.mix * np.abs(audio)

        return result


class HarmonicsEffect(AudioEffect):
    """Add harmonic overtones to the signal."""

    def __init__(self, harmonics: list[tuple[int, float]]) -> None:
        """Initialize harmonics effect.

        Args:
            harmonics: List of (harmonic_number, amplitude) tuples,
                      e.g. [(2, 0.3), (3, 0.15)] adds 2nd and 3rd harmonics
        """
        self.harmonics = harmonics

    def apply(
        self, audio: np.ndarray, t: np.ndarray, frequency: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Add harmonics modulated by input amplitude."""
        result = audio.copy()
        audio_amplitude = np.abs(audio)

        for harmonic_num, amplitude in self.harmonics:
            harmonic_freq = frequency * harmonic_num
            phase = np.cumsum(2 * np.pi * harmonic_freq / sample_rate)
            result += amplitude * np.sin(phase) * audio_amplitude

        return result


class DistortionEffect(AudioEffect):
    """Add soft clipping distortion."""

    def __init__(self, gain: float = 2.0, threshold: float = 0.7) -> None:
        """Initialize distortion effect.

        Args:
            gain: Gain before clipping (default: 2.0)
            threshold: Clipping threshold 0.0-1.0 (default: 0.7)
        """
        self.gain = gain
        self.threshold = threshold

    def apply(
        self, audio: np.ndarray, t: np.ndarray, frequency: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Apply soft clipping using tanh."""
        distorted = audio * self.gain
        return np.tanh(distorted / self.threshold) * self.threshold


class EchoEffect(AudioEffect):
    """Add echo/delay to the signal."""

    def __init__(
        self, delay_time: float = 0.1, decay: float = 0.5, repeats: int = 3
    ) -> None:
        """Initialize echo effect.

        Args:
            delay_time: Delay in seconds (default: 0.1)
            decay: Decay factor 0.0-1.0 (default: 0.5)
            repeats: Number of echoes (default: 3)
        """
        self.delay_time = delay_time
        self.decay = decay
        self.repeats = repeats

    def apply(
        self, audio: np.ndarray, t: np.ndarray, frequency: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Apply echo effect."""
        result = audio.copy()
        delay_samples = int(self.delay_time * sample_rate)

        for i in range(1, self.repeats + 1):
            offset = delay_samples * i
            if offset < len(audio):
                result[offset:] += audio[:-offset] * (self.decay**i)

        return result


class WahWahEffect(AudioEffect):
    """Add sweeping filter effect (wah-wah)."""

    def __init__(
        self,
        min_freq: float = 200,
        max_freq: float = 2000,
        sweep_rate: float = 1.0,
        resonance: float = 5.0,
    ) -> None:
        """Initialize wah-wah effect.

        Args:
            min_freq: Minimum filter frequency in Hz (default: 200)
            max_freq: Maximum filter frequency in Hz (default: 2000)
            sweep_rate: Sweep rate in Hz (default: 1.0)
            resonance: Filter resonance/Q factor (default: 5.0)
        """
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sweep_rate = sweep_rate
        self.resonance = resonance

    def apply(
        self, audio: np.ndarray, t: np.ndarray, frequency: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Apply simplified wah-wah filter."""
        sweep_freq = self.min_freq + (self.max_freq - self.min_freq) * (
            0.5 + 0.5 * np.sin(2 * np.pi * self.sweep_rate * t)
        )
        # Scale modulation depth based on signal amplitude to preserve fades
        # Modulation ranges from 0.85 to 1.0 (no boost above original level)
        signal_envelope = np.abs(audio) / (np.max(np.abs(audio)) + 1e-10)
        modulation_depth = 0.15 * signal_envelope
        sweep_modulation = np.sin(2 * np.pi * sweep_freq * t / sample_rate * 100)
        modulation = 1.0 - modulation_depth + modulation_depth * sweep_modulation
        return audio * modulation


class TremoloEffect(AudioEffect):
    """Add amplitude modulation (tremolo)."""

    def __init__(self, rate: float = 5.0, depth: float = 0.5) -> None:
        """Initialize tremolo effect.

        Args:
            rate: Modulation rate in Hz (default: 5.0)
            depth: Modulation depth 0.0-1.0 (default: 0.5)
        """
        self.rate = rate
        self.depth = depth

    def apply(
        self, audio: np.ndarray, t: np.ndarray, frequency: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Apply tremolo effect."""
        modulation = 1 - self.depth * (0.5 + 0.5 * np.sin(2 * np.pi * self.rate * t))
        return audio * modulation


class PulseEffect(AudioEffect):
    """Create pulsing/beeping pattern by gating audio on and off."""

    def __init__(
        self,
        pulse_rate: float = 4.0,
        duty_cycle: float = 0.5,
        fade_time: float = 0.01,
    ) -> None:
        """Initialize pulse effect.

        Args:
            pulse_rate: Pulses per second (default: 4.0)
            duty_cycle: Fraction of time on 0.0-1.0 (default: 0.5)
            fade_time: Transition time in seconds (default: 0.01)
        """
        self.pulse_rate = pulse_rate
        self.duty_cycle = duty_cycle
        self.fade_time = fade_time

    def apply(
        self, audio: np.ndarray, t: np.ndarray, frequency: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Apply pulsing pattern with smooth transitions."""
        pulse_period = 1.0 / self.pulse_rate
        on_time = pulse_period * self.duty_cycle

        gate = np.zeros_like(t)
        for i, time in enumerate(t):
            phase = (time % pulse_period) / pulse_period

            if phase < self.duty_cycle:
                time_in_on = time % pulse_period
                if time_in_on < self.fade_time:
                    gate[i] = time_in_on / self.fade_time
                elif time_in_on > (on_time - self.fade_time):
                    time_to_end = on_time - time_in_on
                    gate[i] = time_to_end / self.fade_time
                else:
                    gate[i] = 1.0

        return audio * gate


class NoiseEffect(AudioEffect):
    """Add filtered noise for breath/texture."""

    def __init__(self, amount: float = 0.1, high_pass_freq: float = 500) -> None:
        """Initialize noise effect.

        Args:
            amount: Noise amount 0.0-1.0 (default: 0.1)
            high_pass_freq: High-pass filter frequency in Hz (default: 500)
        """
        self.amount = amount
        self.high_pass_freq = high_pass_freq

    def apply(
        self, audio: np.ndarray, t: np.ndarray, frequency: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Add high-pass filtered white noise scaled by audio amplitude."""
        noise = np.random.uniform(-1, 1, len(audio))

        alpha = self.high_pass_freq / (self.high_pass_freq + sample_rate / (2 * np.pi))
        filtered_noise = np.zeros_like(noise)
        filtered_noise[0] = noise[0]
        for i in range(1, len(noise)):
            filtered_noise[i] = alpha * (
                filtered_noise[i - 1] + noise[i] - noise[i - 1]
            )

        audio_amplitude = np.abs(audio)
        scaled_noise = filtered_noise * audio_amplitude * self.amount

        return audio + scaled_noise


class PitchBendEffect(AudioEffect):
    """Bend pitch up or down over time."""

    def __init__(self, bend_amount: float = 0.2, bend_rate: float = 2.0) -> None:
        """Initialize pitch bend effect.

        Args:
            bend_amount: Maximum pitch deviation in semitones (default: 0.2)
            bend_rate: Modulation speed in Hz (default: 2.0)
        """
        self.bend_amount = bend_amount
        self.bend_rate = bend_rate

    def apply(
        self, audio: np.ndarray, t: np.ndarray, frequency: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Apply pitch bend via frequency modulation without time shift."""
        max_ratio = 2 ** (self.bend_amount / 12)
        bend = 1 + (max_ratio - 1) * np.sin(2 * np.pi * self.bend_rate * t)

        # Apply instantaneous frequency modulation
        bent_freq = frequency * bend
        phase = np.cumsum(2 * np.pi * bent_freq / sample_rate)

        # Preserve amplitude envelope by multiplying with normalized original
        envelope = np.abs(audio) / (np.abs(audio).max() + 1e-10)
        modulated = np.sin(phase) * envelope * (np.abs(audio).max())

        return modulated


class BitCrushEffect(AudioEffect):
    """Reduce bit depth for lo-fi/digital effect."""

    def __init__(self, bit_depth: int = 4) -> None:
        """Initialize bit crush effect.

        Args:
            bit_depth: Target bit depth 1-16 (default: 4)
        """
        self.bit_depth = max(1, min(16, bit_depth))

    def apply(
        self, audio: np.ndarray, t: np.ndarray, frequency: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Apply bit depth reduction."""
        levels = 2**self.bit_depth
        return np.round(audio * levels) / levels


class StutterEffect(AudioEffect):
    """Create stuttering/glitchy effect."""

    def __init__(
        self, stutter_rate: float = 16.0, stutter_length: float = 0.03
    ) -> None:
        """Initialize stutter effect.

        Args:
            stutter_rate: Stutters per second (default: 16.0)
            stutter_length: Length of each stutter segment in seconds (default: 0.03)
        """
        self.stutter_rate = stutter_rate
        self.stutter_length = stutter_length

    def apply(
        self, audio: np.ndarray, t: np.ndarray, frequency: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Apply stuttering by repeating small segments."""
        result = audio.copy()
        stutter_period = 1.0 / self.stutter_rate
        stutter_samples = int(self.stutter_length * sample_rate)

        for i in range(int(t[-1] * self.stutter_rate)):
            stutter_time = i * stutter_period
            stutter_idx = int(stutter_time * sample_rate)

            if stutter_idx < len(audio) and stutter_idx + stutter_samples < len(audio):
                segment = audio[stutter_idx : stutter_idx + stutter_samples].copy()
                result[stutter_idx : stutter_idx + stutter_samples] = segment

        return result


class SmoothingEffect(AudioEffect):
    """Smooth out rapid amplitude changes using low-pass filter.

    Useful when audio follows animation velocity too closely, creating
    a jittery quality. Applies moving average to smooth transitions.
    """

    def __init__(self, smoothing_time: float = 0.05) -> None:
        """Initialize smoothing effect.

        Args:
            smoothing_time: Smoothing window in seconds, typical: 0.02-0.1
                (default: 0.05)
        """
        self.smoothing_time = smoothing_time

    def apply(
        self, audio: np.ndarray, t: np.ndarray, frequency: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Apply moving average low-pass filter."""
        window_samples = int(self.smoothing_time * sample_rate)
        if window_samples < 1:
            return audio

        if window_samples % 2 == 0:
            window_samples += 1

        kernel = np.ones(window_samples) / window_samples
        return np.convolve(audio, kernel, mode="same")
