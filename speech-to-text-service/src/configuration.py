# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
from workmesh.config import BaseConfig

type Seconds = float


class AudioConfig(BaseModel):
    sample_rate: int = Field(
        default=16000,
        description="The sample rate of the audio input. Common sample rates are: "
        "16000, 24000, 32000, 44100, 48000",
    )
    channels: int = Field(
        default=1, description="The number of channels of the audio input"
    )
    chunk_duration: Seconds = Field(
        default=0.025, description="Duration of audio chunks in seconds"
    )


class EndpointingConfig(BaseModel):
    start_history: int = Field(
        default=300,
        description="""Size of the window, in milliseconds, to use to detect start of
        utterance. If (start_th) of (start_history) ms of the acoustic model output
        have non-blank tokens, start of utterance is detected""",
    )
    start_threshold: float = Field(
        default=0.2,
        le=1.0,
        gt=0,
        description="""Percentage threshold to use to detect start of utterance.
        If (start_th) of (start_history) ms of the acoustic model output have non-blank
        tokens, start of utterance is detected""",
    )
    stop_history: int = Field(
        default=1500,
        description="""Size of the window, in milliseconds, to use to detect end of
        utterance. If (stop_th) of (stop_history) ms of the acoustic model output have
        non-blank tokens, end of utterance is detected and decoder will be reset""",
    )
    stop_history_eou: int = Field(
        default=1500,
        description="""Size of the window, in milliseconds, to trigger end of utterance
        first pass. If (stop_th_eou) of (stop_history_eou) ms of the acoustic model
        output have non-blank tokens, a partial transcript with high stability
        will be generated""",
    )
    stop_threshold: float = Field(
        default=0.98,
        le=1.0,
        gt=0,
        description="""Percentage threshold to use to detect end of utterance.
        If (stop_th) of (stop_history) ms of the acoustic model output have non-blank
        tokens, end of utterance is detected""",
    )
    stop_threshold_eou: float = Field(
        default=0.98,
        le=1.0,
        gt=0,
        description="""Percentage threshold to use to detect end of utterance.
        If (stop_th_eou) of (stop_history_eou) ms of the acoustic model output
        have non-blank tokens, end of utterance for the first pass will be triggered""",
    )


class VadConfig(BaseModel):
    onset: float = Field(
        default=0.85,
        description="Onset threshold for detecting the beginning and end of a speech",
    )
    offset: float = Field(
        default=0.85, description="Offset threshold for detecting the end of a speech"
    )
    min_duration_on: float = Field(
        default=0.2, description="Threshold for small non_speech deletion"
    )
    min_duration_off: float = Field(
        default=0.5, description="Threshold for short speech segment deletion"
    )
    pad_offset: float = Field(
        default=0.08, description="Add durations after each speech segment"
    )
    pad_onset: float = Field(
        default=0.3, description="Add durations before each speech segment"
    )


class RivaConfig(BaseModel):
    url: str = Field(default="riva:50051", description="The URL of the Riva server")
    model: str = Field(
        default="parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer",
        description="The model to use for the Riva server",
    )
    language_code: str = Field(
        default="en-US", description="The language code to use for the Riva server"
    )
    profanity_filter: bool = Field(
        default=False, description="Whether to filter profanity from the transcript"
    )
    enable_automatic_punctuation: bool = Field(
        default=True, description="Whether to enable automatic punctuation"
    )
    verbatim_transcripts: bool = Field(
        default=False, description="Whether to enable verbatim transcripts"
    )
    endpointing: EndpointingConfig = Field(default=EndpointingConfig())
    vad: VadConfig = Field(default=VadConfig())


class SpeechToTextConfig(BaseConfig):
    input_devices_index_or_name: list[int | str] = Field(
        default=["Reachy Mini Audio", "reSpeaker"]
    )
    audio_config: AudioConfig = Field(default=AudioConfig())
    riva_config: RivaConfig = Field(default=RivaConfig())
