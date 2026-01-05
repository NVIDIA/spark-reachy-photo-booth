# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import asyncio
from collections.abc import AsyncGenerator

from configuration import TextToSpeechServiceConfig
from models.base_tts_model import BaseTTSModel
from models.kokoro_tts import KokoroTTSModel
from workmesh.config import load_config
from workmesh.messages import Audio, ClipData, HumanSpeechRequest
from workmesh.service import Service, produces, subscribe
from workmesh.service_executor import ServiceExecutor

from workmesh import clip_data_topic, human_speech_request_topic


class TextToSpeechService(Service):
    def __init__(self, config: TextToSpeechServiceConfig) -> None:
        super().__init__(config)
        self.config = config

        # Load TTS model
        self.logger.info(f"Loading TTS model: {self.config.tts_model_config}")
        self.tts_model = self._load_tts_model()
        self.logger.info("TTS model loaded.")

    @subscribe(human_speech_request_topic)
    @produces(clip_data_topic)
    async def on_human_speech_request(
        self, message: HumanSpeechRequest
    ) -> AsyncGenerator[ClipData, None]:
        self.logger.info(
            "Human speech request received. "
            + f"Robot ID: {message.robot_id} Script: '{message.script}'"
        )

        try:
            # Filter text
            filtered_text = self._filter_text(message.script)
            self.logger.info(f"Filtered text: {filtered_text}")

            # Generate audio
            voice_data = self.tts_model.generate_audio(filtered_text)

            if voice_data is None:
                self.logger.error("Failed to generate audio")
                return
            self.logger.info("Audio generated successfully!")

            self.logger.info(f"Sending audio with uuid: {message.action_uuid}")
            yield ClipData(
                action_uuid=message.action_uuid,
                robot_id=message.robot_id,
                audio=Audio(
                    sample_rate=self.config.audio_config.sample_rate,
                    bits_per_sample=self.config.audio_config.bits_per_sample,
                    channel_count=self.config.audio_config.channel_count,
                    audio_buffer=voice_data,
                ),
            )
            self.logger.info("Audio sent.")
        except Exception as e:
            self.logger.error(f"TTS model failed to generate audio: {e}")
            return

    def _load_tts_model(self) -> BaseTTSModel:
        # NOTE: we can add more models and engines here
        if self.config.tts_model_config.engine == "Kokoro":
            return KokoroTTSModel(self.config, self.logger)
        else:
            raise ValueError(
                f"Model {self.config.tts_model_config.engine} not supported"
            )

    def _filter_text(self, text: str) -> str:
        text = text.strip().lstrip("'").rstrip("'")
        try:
            parsed_text = ast.literal_eval(text)
            if isinstance(parsed_text, dict) and "prompt" in parsed_text:
                text = parsed_text["prompt"]
        except Exception:
            # Use original text
            self.logger.debug("Text parsing as literal failed: Using original text")

        return text.strip().replace("\n", " ").replace("\t", " ")

    async def stop(self) -> None:
        self.logger.info("Shutting down Text-To-Speech service!")
        await super().stop()


async def main() -> None:
    config = load_config(TextToSpeechServiceConfig)
    await ServiceExecutor([TextToSpeechService(config)]).run()


if __name__ == "__main__":
    asyncio.run(main())
