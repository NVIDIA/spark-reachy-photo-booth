# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field
from pydantic.networks import IPv4Address  # type: ignore
from workmesh.config import BaseConfig


class UiServerConfig(BaseConfig):
    """Configuration for the overlay renderer."""

    port: int = Field(default=9000, description="The port to run the webserver on.")
    host: IPv4Address = Field(
        default=IPv4Address("0.0.0.0"),
        description="The host to run the webserver on.",
    )
    minio_public_host: str = Field(
        default="127.0.0.1",
        description="The publicly accesible hostname/ip for the minio instances.",
    )
