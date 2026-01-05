#!/usr/bin/env uv run
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pydub",
# ]
# ///
"""Audio utility functions."""

import os
import sys
import wave

import numpy as np
from configuration import DEFAULT_SAMPLE_RATE


def export_audio(
    audio: np.ndarray, filename: str, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> None:
    """Export audio as 16-bit mono WAV file.

    Args:
        audio: Audio signal as numpy array (values -1.0 to 1.0)
        filename: Output WAV file path
        sample_rate: Sample rate in Hz (default: from configuration module)
    """
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


def play_audio(filename: str) -> None:
    """Play WAV file.

    Args:
        filename: Path to WAV file to play

    Note:
        Requires pydub and simpleaudio: pip install pydub simpleaudio
    """
    # On Windows, use os.startfile as fallback to avoid temp file issues
    if sys.platform == "win32":
        try:
            os.startfile(filename)
            return
        except Exception:
            pass

    # Try pydub/simpleaudio on other platforms or as fallback
    try:
        from pydub import AudioSegment
        from pydub.playback import play

        audio = AudioSegment.from_wav(filename)
        play(audio)
    except ImportError as e:
        raise ImportError("Install pydub and simpleaudio: pip install pydub") from e
