# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod

import numpy as np
import pydub
import torch
from configuration import TextToSpeechServiceConfig


class BaseTTSModel(ABC):
    def __init__(
        self,
        config: TextToSpeechServiceConfig,
        logger: logging.Logger,
    ):
        self.config = config
        self._logger = logger

        # Initialize device
        if torch.cuda.is_available() and self.config.device == "cuda":
            self.device = "cuda"
            self._logger.info("CUDA is available. Using GPU for TTS model.")
        else:
            self.device = "cpu"
            self._logger.info("Using CPU for TTS model.")

    @abstractmethod
    def generate_audio(self, text: str) -> bytes | None:
        raise NotImplementedError("Method not implemented")

    def convert_audio(
        self,
        audio: np.ndarray,
        sample_rate_src: int,
        sample_width_src: int,
        channel_count_src: int,
    ) -> bytes | None:
        converted_audio: pydub.AudioSegment = pydub.AudioSegment(
            audio.tobytes(),
            sample_width=sample_width_src,
            frame_rate=sample_rate_src,
            channels=channel_count_src,
        )

        converted_audio = (
            converted_audio.set_frame_rate(self.config.audio_config.sample_rate)
            .set_channels(self.config.audio_config.channel_count)
            .set_sample_width(self.config.audio_config.bits_per_sample // 8)
        )

        if converted_audio.raw_data is None:
            self._logger.error("Failed to convert audio to the desired format")
            return None

        return converted_audio.raw_data
