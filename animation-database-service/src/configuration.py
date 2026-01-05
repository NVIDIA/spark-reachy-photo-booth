# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Self

SAMPLE_RATES = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000]
BITS_PER_SAMPLE = [8, 16, 24, 32]
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_BITS_PER_SAMPLE = 16

# Try to import pydantic-dependent classes (not available in Blender)
try:
    from pydantic import BaseModel, DirectoryPath, Field, model_validator
    from workmesh.config import BaseConfig

    class AudioConfig(BaseModel):
        sample_rate: int = Field(ge=1, lt=100000, default=DEFAULT_SAMPLE_RATE)
        bits_per_sample: int = Field(ge=1, lt=100, default=DEFAULT_BITS_PER_SAMPLE)
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
                    f"Wrong sample_rate: {self.sample_rate}. "
                    f"Must be one of {SAMPLE_RATES}"
                )
            return self

    class DatabaseConfig(BaseConfig):
        clip_directory: DirectoryPath = Path("/app/assets/animLibrary")
        frame_rate: int = Field(ge=1, default=30)
        audio_config: AudioConfig = AudioConfig()
        max_request_size: int = Field(
            gt=0, default=104857600
        )  # 100MB - Note: requires workmesh package rebuild
        neck_offset_factor: float = Field(
            ge=0.0,
            le=1.0,
            default=0.2,
            description="The factor to apply to the bounding box width \
                to get the neck offset.",
        )

except ImportError:
    # Pydantic not available (e.g. in Blender) - config classes won't be available
    # but constants (DEFAULT_SAMPLE_RATE, etc.) are still accessible
    pass
