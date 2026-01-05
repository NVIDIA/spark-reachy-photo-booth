# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import threading
import time
from copy import deepcopy

import numpy as np
import pydub
import sounddevice as sd
from clips.base_clips import BaseClip
from clips.static_clips import StaticClip
from configuration import AudioConfig
from utils import find_output_device_index, get_format_from_width


class AudioMixer:
    """Audio mixer."""

    def __init__(
        self, frame_rate: int, config: AudioConfig, logger: logging.Logger
    ) -> None:
        # Clips
        self.audio_clips: dict[str, BaseClip] = {}
        self.audio_clips_lock: threading.Lock = threading.Lock()

        # Configs
        self.logger: logging.Logger = logger
        self.config: AudioConfig = config
        self.frame_rate: int = frame_rate
        self.dtype: np.dtype = get_format_from_width(self.config.bits_per_sample // 8)

        # Play task
        self.stop_event: asyncio.Event = asyncio.Event()
        self.play_audio_clips_task: asyncio.Task = asyncio.create_task(
            self._play_audio_clips()
        )
        self.logger.info(
            f"Audio mixer initialized with the following config: {self.config}"
        )

        # Find output device
        found, self.output_device_index, output_device_name = find_output_device_index(
            self.config.output_devices_index_or_name
        )
        if not found:
            self.logger.warning(
                f"Output device {self.config.output_devices_index_or_name} not found, "
                + "using default device"
            )
        self.logger.info(
            f"Using output device {output_device_name} ({self.output_device_index})"  # noqa: E501
        )

    async def close(self) -> None:
        """Close the audio mixer."""

        # Stop playing audio clips
        self.stop_event.set()
        await self.play_audio_clips_task

        self.logger.info("Audio mixer closed.")

    async def play(self, clip: BaseClip) -> None:
        """Play an audio clip."""

        assert clip.audio

        if clip.audio.audio_data.raw_data is None:
            self.logger.warning(
                f"Can't play clip {clip.action_uuid}. Audio data is not available."
            )
            return

        with self.audio_clips_lock:
            self.audio_clips[clip.action_uuid] = deepcopy(clip)

        self.logger.debug(f"Added clip {clip.action_uuid} to the audio mixer.")

    def stop(self, action_uuid: str) -> None:
        """Stop a specific audio clip."""

        with self.audio_clips_lock:
            self.audio_clips.pop(action_uuid)
        self.logger.debug(f"Removed clip {action_uuid} from the audio mixer.")

    def fade_out_clip(self, action_uuid: str, fade_out: float = 0.0) -> None:
        """Fade out a specific audio clip."""

        with self.audio_clips_lock:
            clip = self.audio_clips[action_uuid]
            assert clip.audio and clip.start_time

            # Calculate how much time has passed since the clip started
            fade_out = int(fade_out * 1000)
            current_audio_time = int((time.time() - clip.start_time) * 1000)
            cut_clip = clip.audio.audio_data[: current_audio_time + fade_out]
            clip.audio.audio_data = cut_clip.fade_out(fade_out)  # type: ignore

        self.logger.debug(f"Fading out clip {action_uuid}.")

    def change_volume(self, action_uuid: str, volume: float) -> bool:
        """Change the volume of a specific audio clip."""
        with self.audio_clips_lock:
            clip = self.audio_clips.get(action_uuid)
            if clip is None:
                self.logger.warning(
                    "Can't change volume. "
                    + f"Clip {action_uuid} not found in the audio mixer."
                )
                return False

            # We should only have audio clips in the audio mixer
            assert clip.audio
            clip.audio.change_volume(volume)
            self.logger.debug(f"Changed volume of clip {action_uuid} to {volume}.")

            return True

    async def _play_audio_clips(self) -> None:
        """Play audio clips."""

        # Define callback for playback
        def audio_callback(
            outdata: np.ndarray,
            frame_count: int,
            _time,
            _status: sd.CallbackFlags,
        ) -> None:
            audio_chunks = []

            # Calculate chunk size
            chunk_size = (
                frame_count
                * self.config.bits_per_sample
                // 8
                * self.config.channel_count  # noqa: E501
            )

            with self.audio_clips_lock:
                for clip in self.audio_clips.values():
                    assert clip.audio is not None
                    assert clip.audio.audio_data.raw_data is not None

                    n_bytes = chunk_size
                    if clip.audio.n_bytes_read + n_bytes >= len(
                        clip.audio.audio_data.raw_data
                    ):
                        n_bytes = (
                            len(clip.audio.audio_data.raw_data)
                            - clip.audio.n_bytes_read
                        )

                    # Extract chunk from audio clip
                    chunk = clip.audio.audio_data.raw_data[
                        clip.audio.n_bytes_read : clip.audio.n_bytes_read + n_bytes
                    ]
                    clip.audio.n_bytes_read += n_bytes

                    # If clip is looping, add the remaining bytes to the chunk
                    if (
                        isinstance(clip, StaticClip)
                        and clip.loop
                        and n_bytes < chunk_size
                    ):
                        left_to_read = chunk_size - n_bytes
                        chunk += clip.audio.audio_data.raw_data[:left_to_read]
                        clip.audio.n_bytes_read = left_to_read
                        n_bytes = chunk_size

                    # Add to chunks to mix only if the clip is not finished
                    if n_bytes > 0:
                        audio_chunks.append(chunk)

            # Mix chunks
            duration_float = frame_count / self.config.sample_rate * 1000
            duration_ms: int = int(duration_float) + (
                1 if not duration_float.is_integer() else 0
            )
            audio = pydub.AudioSegment.silent(
                duration=duration_ms, frame_rate=self.config.sample_rate
            )

            for chunk in audio_chunks:
                audio_chunk = pydub.AudioSegment(
                    chunk,
                    sample_width=self.config.bits_per_sample // 8,
                    frame_rate=self.config.sample_rate,
                    channels=self.config.channel_count,
                )
                audio = audio.overlay(audio_chunk)

            data: bytes | None = audio.raw_data
            if data is None:
                self.logger.warning(
                    "Can't play audio clips. Audio data is not available."
                )
                data = b""

            # Transform data to numpy array
            audio_array = np.frombuffer(data, dtype=self.dtype)
            audio_array = audio_array.reshape(-1, self.config.channel_count)

            # Reshape audio array for output
            if audio_array.shape[0] >= frame_count:
                audio_array = audio_array[:frame_count]
            else:
                self.logger.warning(
                    f"Audio array has fewer frames ({audio_array.shape[0]}) than "
                    f"expected ({frame_count}). There might be an issue with the mixing."  # noqa: E501
                )

            # Write to output data
            outdata[:] = audio_array

        try:
            with sd.OutputStream(
                callback=audio_callback,
                channels=self.config.channel_count,
                samplerate=self.config.sample_rate,
                device=self.output_device_index,
                dtype=self.dtype,
            ) as stream:
                while stream.active and not self.stop_event.is_set():
                    await asyncio.sleep(0.1)

            self.logger.debug("Stream closed.")

        except Exception as e:
            self.logger.exception(f"Error when playing audio clips: {e}")
