# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import pathlib
from enum import Enum

import yaml
from deepmerge import always_merger
from pydantic import BaseModel, field_validator
from pydantic.networks import HttpUrl, KafkaDsn

LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def _default_broker_url():
    # Check for /.dockerenv as a common indicator of running in Docker
    if os.path.exists("/.dockerenv"):
        return "kafka://redpanda:9092"
    return "kafka://localhost:19092"


class OffsetType(str, Enum):
    earliest = "earliest"
    latest = "latest"


class ProducerConfig(BaseModel):
    broker_url: KafkaDsn = KafkaDsn(_default_broker_url())
    max_request_size: int = 1048576  # Default to 1MB


class ConsumerConfig(BaseModel):
    broker_url: KafkaDsn = KafkaDsn(_default_broker_url())
    consumer_group: str | None = None
    offset_type: OffsetType = OffsetType.latest
    enable_auto_commit: bool = True


class BaseConfig(ConsumerConfig, ProducerConfig):
    otel_endpoint: HttpUrl = HttpUrl("http://lgtm-otel:4317")
    log_level: str = "INFO"

    @field_validator("log_level", mode="after")
    @classmethod
    def log_level_validator(cls, value: str) -> str:
        if LOG_LEVELS.get(value.upper()) is None:
            raise ValueError(
                f"Invalid log level: {value}. Valid values are {LOG_LEVELS}"
            )
        return value.upper()


def load_config[T: BaseModel](cls: type[T]):
    try:
        yaml_contents = yaml.safe_load(pathlib.Path("/config.yaml").read_text())
        print(yaml_contents)
        parsed: T = cls(**yaml_contents)
        default: T = cls()
        parsed_dict = parsed.model_dump(exclude_unset=True)
        default_dict = default.model_dump(exclude_unset=True)
        merged_dict = always_merger.merge(default_dict, parsed_dict)
        return cls.model_validate(merged_dict)
    except FileNotFoundError:
        return cls()
