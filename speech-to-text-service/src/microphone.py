# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import queue

import riva
import riva.client

try:
    import riva.client.audio_io
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    print("Please install pyaudio from https://pypi.org/project/PyAudio")
    exit(1)


class PausableRivaMicrophoneStream(riva.client.audio_io.MicrophoneStream):
    """
    Extended Riva MicrophoneStream that provides pause/resume functionality.
    """

    def __init__(self, rate: int, chunk: int, device: int | None = None) -> None:
        super().__init__(rate, chunk, device)  # type: ignore[arg-type]
        self._is_paused = False

    def pause(self) -> None:
        """
        Pause the audio stream and clear any buffered audio data.

        This method stops the PyAudio stream from capturing new audio and discards
        any audio data that was already buffered, ensuring a clean resume.
        """
        if (
            not self._is_paused
            and hasattr(self, "_audio_stream")
            and not self._audio_stream.is_stopped()
        ):
            self._audio_stream.stop_stream()
            self._clear_buffer()
            self._is_paused = True

    def resume(self) -> None:
        """
        Resume the audio stream.

        This restarts the PyAudio stream to begin capturing audio again.
        """
        if (
            self._is_paused
            and hasattr(self, "_audio_stream")
            and self._audio_stream.is_stopped()
        ):
            self._audio_stream.start_stream()
            self._is_paused = False

    def is_paused(self) -> bool:
        """Check if the audio stream is currently paused."""
        return self._is_paused

    def _clear_buffer(self) -> None:
        """
        Clear the audio buffer to discard accumulated audio.

        This prevents old audio data from being processed after resuming.
        """
        if hasattr(self, "_buff"):
            while not self._buff.empty():
                try:
                    self._buff.get_nowait()
                except queue.Empty:
                    break
