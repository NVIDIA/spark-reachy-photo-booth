# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
import time
from collections.abc import AsyncGenerator

import riva
import riva.client
from configuration import SpeechToTextConfig
from grpc import RpcError
from microphone import PausableRivaMicrophoneStream
from utils import Transcription, find_device_index
from workmesh.config import load_config
from workmesh.messages import (
    Command,
    ServiceCommand,
    UserUtterance,
    UserUtteranceStatus,
)
from workmesh.messages import (
    Service as ServiceName,
)
from workmesh.service import Service
from workmesh.service_executor import ServiceExecutor

from workmesh import produces, service_command_topic, subscribe, user_utterance_topic


class SpeechToTextService(Service):
    def __init__(self, config: SpeechToTextConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

        # Find audio input device
        found, device_index, device_name = find_device_index(
            config.input_devices_index_or_name
        )

        if not found:
            self.logger.warning(
                f"Device {config.input_devices_index_or_name} not found, using default device"  # noqa: E501
            )
        self.device_index = device_index
        self.logger.info(f'Using device "{device_name}" ({device_index})')

        # Service status
        self.service_enabled: threading.Event = threading.Event()
        self.service_enabled.set()  # Start the service enabled by default

        # Utterance message queue
        self.message_queue: asyncio.Queue[UserUtterance | None] = asyncio.Queue()

        # Start the producer
        self.create_task(self.produce_message())  # type: ignore

        # Transcribe worker
        self.stop_event = threading.Event()
        self.running_loop = asyncio.get_running_loop()
        self.transcribe_task = asyncio.get_running_loop().run_in_executor(
            None, self.transcribe_worker
        )

    @subscribe(service_command_topic)
    async def on_service_command(self, service_command: ServiceCommand) -> None:
        if service_command.target_service == ServiceName.STT:
            if (
                service_command.command == Command.ENABLE
                and not self.service_enabled.is_set()
            ):
                self.service_enabled.set()
                self.logger.info("STT service enabled.")
            elif (
                service_command.command == Command.DISABLE
                and self.service_enabled.is_set()
            ):
                self.service_enabled.clear()
                self.logger.info("STT service disabled.")

    @produces(user_utterance_topic)
    async def produce_message(self) -> AsyncGenerator[UserUtterance, None]:
        self.logger.info("Producing message started.")

        try:
            while True:
                message = await self.message_queue.get()

                # Stop the producer
                if message is None:
                    break

                yield message
            self.logger.info("Producing message stopped.")
        except Exception as e:
            self.logger.exception(f"Error producing message: {e}")
        except asyncio.CancelledError:
            self.logger.info("Producing message was cancelled.")

    def transcribe_worker(self) -> None:
        """Worker thread for transcribing audio chunks."""

        self.logger.info("Transcription worker started.")

        try:
            # Riva ASR service
            self.logger.info(
                f"Connecting to Riva server at {self.config.riva_config.url}"
            )
            auth = riva.client.Auth(use_ssl=False, uri=self.config.riva_config.url)
            asr_service = riva.client.ASRService(auth)
            riva_streaming_config = riva.client.StreamingRecognitionConfig(
                config=riva.client.RecognitionConfig(
                    encoding=riva.client.AudioEncoding.LINEAR_PCM,
                    max_alternatives=1,
                    language_code=self.config.riva_config.language_code,
                    model=self.config.riva_config.model,
                    profanity_filter=self.config.riva_config.profanity_filter,
                    enable_automatic_punctuation=self.config.riva_config.enable_automatic_punctuation,
                    verbatim_transcripts=self.config.riva_config.verbatim_transcripts,
                    sample_rate_hertz=self.config.audio_config.sample_rate,
                    audio_channel_count=self.config.audio_config.channels,
                ),
                interim_results=True,
            )

            riva.client.add_endpoint_parameters_to_config(
                riva_streaming_config,
                self.config.riva_config.endpointing.start_history,
                self.config.riva_config.endpointing.start_threshold,
                self.config.riva_config.endpointing.stop_history,
                self.config.riva_config.endpointing.stop_history_eou,
                self.config.riva_config.endpointing.stop_threshold,
                self.config.riva_config.endpointing.stop_threshold_eou,
            )
            riva.client.add_custom_configuration_to_config(
                riva_streaming_config,
                f"neural_vad.onset:{self.config.riva_config.vad.onset},"
                + f"neural_vad.offset:{self.config.riva_config.vad.offset},"
                + "neural_vad.min_duration_on:"
                + f"{self.config.riva_config.vad.min_duration_on},"
                + "neural_vad.min_duration_off:"
                + f"{self.config.riva_config.vad.min_duration_off},"
                + f"neural_vad.pad_offset:{self.config.riva_config.vad.pad_offset},"
                + f"neural_vad.pad_onset:{self.config.riva_config.vad.pad_onset}",
            )
            self.logger.debug(f"Client Config: {riva_streaming_config}")

            with PausableRivaMicrophoneStream(
                self.config.audio_config.sample_rate,
                int(
                    self.config.audio_config.chunk_duration
                    * self.config.audio_config.sample_rate
                ),
                device=self.device_index,
            ) as audio_chunk_iterator:
                self.logger.info("Input stream opened.")

                while not self.stop_event.is_set():
                    current_transcription = Transcription()

                    # If service is disabled, pause the stream and clear buffer
                    if (
                        not self.service_enabled.is_set()
                        and not audio_chunk_iterator.is_paused()
                    ):
                        self.logger.info("Pausing audio stream.")
                        audio_chunk_iterator.pause()

                    # Wait for service to be re-enabled
                    if not self.service_enabled.wait(timeout=1.0):
                        continue

                    if audio_chunk_iterator.is_paused():
                        self.logger.info("Resuming audio stream.")
                        audio_chunk_iterator.resume()

                    streaming_generator = asr_service.streaming_response_generator(
                        audio_chunks=audio_chunk_iterator,
                        streaming_config=riva_streaming_config,
                    )

                    for response in streaming_generator:
                        # Stop the audio stream
                        if (
                            self.stop_event.is_set()
                            or not self.service_enabled.is_set()
                        ):
                            break

                        # Skip empty responses
                        if response and len(response.results) == 0:
                            continue

                        # Send started event
                        timestamp = int(time.time() * 1000)

                        if not current_transcription.started:
                            self._send_user_utterance(
                                UserUtterance(
                                    action_uuid=current_transcription.action_uuid,
                                    status=UserUtteranceStatus.USER_UTTERANCE_STARTED,
                                    timestamp=timestamp,
                                    text=current_transcription.transcript,
                                )
                            )

                            current_transcription.started = True
                            self.logger.info(
                                "[TRANSCRIPTION] action_uuid: "
                                f"{current_transcription.action_uuid[:6]}... "
                                "status: STARTED"
                            )

                        transcript = response.results[0].alternatives[0].transcript
                        is_final = response.results[0].is_final

                        status = (
                            UserUtteranceStatus.USER_UTTERANCE_FINISHED
                            if is_final
                            else UserUtteranceStatus.USER_UTTERANCE_UPDATED
                        )

                        if (
                            transcript == current_transcription.transcript
                            and status == current_transcription.status
                        ):
                            continue

                        current_transcription.transcript = transcript
                        current_transcription.status = status

                        self._send_user_utterance(
                            UserUtterance(
                                action_uuid=current_transcription.action_uuid,
                                status=status,
                                timestamp=timestamp,
                                text=transcript,
                            )
                        )

                        self.logger.info(
                            "[TRANSCRIPTION] "
                            + f"action_uuid: {current_transcription.action_uuid[:6]}... "  # noqa: E501
                            + f"status: {'PARTIAL' if not is_final else '=FINAL='} "
                            + f"transcript: {transcript} "
                        )

                        # Reset everything
                        if is_final:
                            current_transcription = Transcription()

                self.logger.info("Input stream closed.")
            self.logger.info("Transcribing worker stopped.")
        except RpcError as e:
            self.logger.error(
                "Error transcribing audio chunks. "
                + f"Code: {e.code()}. Details: {e.details()}"
            )
        except Exception as e:
            self.logger.exception(f"Error transcribing audio chunks: {e}")

    def _send_user_utterance(self, user_utterance: UserUtterance) -> None:
        asyncio.run_coroutine_threadsafe(
            self.message_queue.put(user_utterance), self.running_loop
        )

    async def stop(self) -> None:
        self.stop_event.set()
        self.message_queue.put_nowait(None)
        await asyncio.sleep(1)
        await super().stop()


async def main() -> None:
    config = load_config(SpeechToTextConfig)
    await ServiceExecutor([SpeechToTextService(config)]).run()


if __name__ == "__main__":
    asyncio.run(main())
