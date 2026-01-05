#!/usr/bin/env uv run
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "scipy",
# ]
# ///
"""Debug/visualization tools for procedural audio.

This module provides tools for visualizing and analyzing generated audio,
including waveform plots and spectrograms.

Requires: matplotlib, scipy
Install with: pip install matplotlib scipy
"""

from typing import TYPE_CHECKING, Union

import numpy as np
from configuration import DEFAULT_SAMPLE_RATE

if TYPE_CHECKING:
    import pydub


def create_audio_debug_plot(
    audio: Union[np.ndarray, "pydub.AudioSegment"],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    title: str = "Audio Analysis",
    save_path: str | None = None,
) -> None:
    """Plot waveform and spectrogram for debugging audio generation.

    Args:
        audio: Audio signal as numpy array or pydub.AudioSegment
        sample_rate: Sample rate in Hz, only used if audio is numpy array
            (default: from configuration module)
        title: Title for the plot
        save_path: Path to save plot image, if None displays interactively

    Example:
        >>> from audio import audio_engine, audio_debug
        >>> gen = audio_engine.FrequencyGenerator(base_freq=100, duration=1.0)
        >>> sound = gen.apply_effects(audio_engine.HarmonicsEffect([(2, 0.3)]))
        >>> audio_debug.create_audio_debug_plot(sound, save_path="plot.png")
    """
    try:
        import matplotlib

        if save_path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib not installed.")
        return

    try:
        import pydub

        if isinstance(audio, pydub.AudioSegment):
            sample_rate = audio.frame_rate
            samples = np.array(audio.get_array_of_samples())
            if audio.sample_width == 2:
                audio = samples.astype(np.float32) / 32768.0
            elif audio.sample_width == 1:
                audio = samples.astype(np.float32) / 128.0
            else:
                audio = samples.astype(np.float32) / 2147483648.0
    except ImportError:
        print("Error: pydub not installed.")
        return

    duration = len(audio) / sample_rate
    time = np.linspace(0, duration, len(audio))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax1.plot(time, audio, linewidth=0.5, color="#2E86AB")
    ax1.set_xlabel("Time (s)", fontsize=10)
    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.set_title("Waveform", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, duration)

    max_amp = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    stats_text = f"Max: {max_amp:.3f}  RMS: {rms:.3f}  Duration: {duration:.2f}s"
    bbox_style = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=bbox_style,
        fontsize=9,
    )

    nperseg = min(2048, len(audio) // 4)
    noverlap = nperseg // 2

    from scipy import signal

    frequencies, times, Sxx = signal.spectrogram(
        audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, scaling="spectrum"
    )

    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    im = ax2.pcolormesh(times, frequencies, Sxx_db, shading="gouraud", cmap="viridis")
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_ylabel("Frequency (Hz)", fontsize=10)
    ax2.set_title("Spectrogram", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, duration)
    ax2.set_ylim(0, min(2000, sample_rate // 2))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)

    cbar_ax = fig.add_axes((0.125, 0.02, 0.775, 0.02))
    cbar = plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Power (dB)", fontsize=9)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()
