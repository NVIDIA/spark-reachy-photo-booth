# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Self

from pydantic import BaseModel, Field, model_validator
from workmesh.config import BaseConfig

SAMPLE_RATES = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000]
BITS_PER_SAMPLE = [8, 16, 24, 32]

# NOTE: we can add more models and voices from other engines here
ENGINES = ["Kokoro"]
VOICES = {
    "Kokoro": [
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_heart",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
        "ef_dora",
        "em_alex",
        "em_santa",
        "ff_siwis",
        "hf_alpha",
        "hf_beta",
        "hm_omega",
        "hm_psi",
        "if_sara",
        "im_nicola",
        "jf_alpha",
        "jf_gongitsune",
        "jf_nezumi",
        "jf_tebukuro",
        "jm_kumo",
        "pf_dora",
        "pm_alex",
        "pm_santa",
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
    ],
}
MODELS = {
    "Kokoro": [
        "hexgrad/Kokoro-82M",
    ],
}


class TTSModelConfig(BaseModel):
    engine: str = Field(default="Kokoro")
    model_name: str = Field(default="hexgrad/Kokoro-82M")
    voice_id: str = Field(default="am_puck")
    speed: float = Field(default=1.0, ge=0.1, le=3.0)
    max_characters: int = Field(
        default=200,
        description="Maximum number of characters of the text to generate audio for. "
        + "If the text is longer than this length, it will be chunked into multiple generations.",  # noqa: E501
    )

    @model_validator(mode="after")
    def tts_config_validator(self) -> Self:
        if self.engine not in ENGINES:
            raise ValueError(f"Invalid engine: {self.engine}. Must be one of {ENGINES}")
        if self.model_name not in MODELS[self.engine]:
            raise ValueError(
                f"Invalid model: {self.model_name}. Must be one of {MODELS[self.engine]}"  # noqa: E501
            )
        if self.voice_id not in VOICES[self.engine]:
            raise ValueError(
                f"Invalid voice: {self.voice_id}. Must be one of {VOICES[self.engine]}"
            )

        return self


class AudioConfig(BaseModel):
    sample_rate: int = Field(ge=1, lt=100000, default=16000)
    bits_per_sample: int = Field(ge=1, lt=100, default=16)
    channel_count: int = Field(ge=1, le=2, default=1)

    @model_validator(mode="after")
    def bits_per_sample_validator(self) -> Self:
        if self.bits_per_sample not in BITS_PER_SAMPLE:
            raise ValueError(
                f"Wrong bits_per_sample: {self.bits_per_sample}. "
                + f"Must be one of {BITS_PER_SAMPLE}"
            )
        return self

    @model_validator(mode="after")
    def sample_rate_validator(self) -> Self:
        if self.sample_rate not in SAMPLE_RATES:
            raise ValueError(
                f"Wrong sample_rate: {self.sample_rate}. Must be one of {SAMPLE_RATES}"
            )
        return self


class TextToSpeechServiceConfig(BaseConfig):
    tts_model_config: TTSModelConfig = Field(default_factory=TTSModelConfig)
    device: str = Field(default="cuda")
    audio_config: AudioConfig = Field(default_factory=AudioConfig)
    max_request_size: int = Field(
        gt=0, default=104857600
    )  # 100MB - Note: requires workmesh package rebuild
