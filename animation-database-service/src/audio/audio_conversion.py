# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pydub
from configuration import DatabaseConfig


def convert_audio(config: DatabaseConfig, logger: logging.Logger) -> None:
    logger.info(
        "Converting audio to WAV and config format: "
        + f"{config.audio_config.sample_rate} Hz, "
        + f"{config.audio_config.channel_count} channels, "
        + f"{config.audio_config.bits_per_sample} bits per sample."
    )

    for clip in config.clip_directory.iterdir():
        if not clip.is_dir():
            continue

        audio_segments: list[pydub.AudioSegment] = []
        audio_duration: float = 0.0
        for audio_file in clip.glob("*.wav"):
            if audio_file.name.endswith("_converted.wav"):
                continue

            # Convert audio to WAV and config format
            try:
                audio_segment: pydub.AudioSegment = pydub.AudioSegment.from_wav(
                    audio_file
                )
                audio_segment = (
                    audio_segment.set_frame_rate(config.audio_config.sample_rate)
                    .set_channels(config.audio_config.channel_count)
                    .set_sample_width(config.audio_config.bits_per_sample // 8)
                )
                audio_duration = max(audio_duration, audio_segment.duration_seconds)
                audio_segments.append(audio_segment)

                logger.info(f"Audio '{audio_file.name}' was converted to WAV.")
            except Exception:
                logger.exception(f"Error when converting audio '{audio_file.name}'.")

        if len(audio_segments) > 0:
            mixed_audio: pydub.AudioSegment = pydub.AudioSegment.silent(
                duration=int(audio_duration * 1000),
                frame_rate=config.audio_config.sample_rate,
            )
            for audio in audio_segments:
                mixed_audio = mixed_audio.overlay(audio)

            mixed_audio.export(clip / f"{clip.name}_converted.wav", format="wav")
            logger.info(f"Audio for '{clip.name}' was mixed.")

    logger.info("Audio conversion complete.")
