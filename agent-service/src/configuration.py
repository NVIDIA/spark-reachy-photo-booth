# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from pydantic import Field, field_validator
from workmesh.config import BaseConfig


class AgentServiceConfig(BaseConfig):
    nat_config: Path = Field(
        description="The path to the NAT agent configuration file.",
        default=Path("configs/photobooth.yml"),
    )
    user_utterance_threshold: int = Field(
        default=1_000,
        description="""
The difference in milliseconds between the timestamp of the user utterance
and the timestamp of the last message in the history.
If the difference is greater than the threshold,
the user utterance is considered invalid and is not processed.""",
        ge=0,
        le=10_000,
    )

    @field_validator("nat_config", mode="after")
    def nat_config_validator(cls, value: Path) -> Path:
        if not value.resolve().exists():
            raise ValueError(f"NAT agent configuration file not found: {value}")
        return value
