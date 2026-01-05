# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from pydantic import BaseModel, Field, SecretStr

POLICY_TEMPLATE = """
{{
    "Version": "2012-10-17",
    "Statement": [
        {{
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::{bucket}/*"
        }}
    ]
}}
"""


class MinioConfig(BaseModel):
    base_url: str = Field(
        description="The base URL of the MinIO service.",
        default="minio:9010" if os.path.exists("/.dockerenv") else "localhost:9010",
    )
    access_key: str = Field(
        description="The access key of the MinIO service.",
        default=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    )
    secret_key: SecretStr = Field(
        description="The secret key of the MinIO service.",
        default=SecretStr(os.getenv("MINIO_SECRET_KEY", "minioadmin")),
    )
    bucket: str = Field(
        description="The name of the MinIO bucket.",
        default="photo-booth-agent",
    )
    secure: bool = Field(
        description="Whether to use HTTPS.",
        default=False,
    )
    create_bucket: bool = Field(
        description="Try to create bucket",
        default=True,
    )
    timeout: float = Field(
        description="The timeout for the MinIO service.",
        default=5.0,
    )
    num_retries: int = Field(
        description="The number of retries for the MinIO service.",
        default=3,
    )
    backoff_factor: float = Field(
        description="The backoff factor for the MinIO service.",
        default=0.2,
    )
