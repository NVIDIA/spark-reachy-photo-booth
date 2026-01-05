# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import logging
import os

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.common import TypedBaseModel
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field

from photo_booth_agent.configs.minio import POLICY_TEMPLATE, MinioConfig
from photo_booth_agent.constant import SERVICE_NAME
from photo_booth_agent.utils import capture_image, initialize_minio, wait_for_ack

logger = logging.getLogger(SERVICE_NAME)


class LookAtHumanToolConfig(FunctionBaseConfig, name="look_at_human"):
    base_url: str = Field(
        description="The base URL of the image generation service.",
        default="http://camera:7071"
        if os.path.exists("/.dockerenv")
        else "http://localhost:7071",
    )
    internal_minio: MinioConfig = Field(
        description="The configuration for the MinIO service.",
        default=MinioConfig(),
    )


class LookAtHumanInput(TypedBaseModel, name="look_at_human"):
    """
    This is the input for the look_at_human function.
    """

    action_uuid: str = Field(
        default="",
        description="""
The action UUID of the tool call.
This field will be set by the framework.
You don't need to set it, just leave it empty.
""",
    )


class LookAtHumanOutput(TypedBaseModel, name="look_at_human"):
    """
    This is the output for the look_at_human function.
    """

    image_url_or_path: str = Field(description="The URL or path of the image captured.")


@register_function(config_type=LookAtHumanToolConfig)
async def look_at_human_tool(config: LookAtHumanToolConfig, builder: Builder):
    try:
        minio = initialize_minio(
            config.internal_minio.base_url,
            config.internal_minio.access_key,
            config.internal_minio.secret_key.get_secret_value(),
            config.internal_minio.bucket,
            policy=POLICY_TEMPLATE.format(bucket=config.internal_minio.bucket),
            secure=config.internal_minio.secure,
            create_bucket=config.internal_minio.create_bucket,
            timeout=config.internal_minio.timeout,
            num_retries=config.internal_minio.num_retries,
            backoff_factor=config.internal_minio.backoff_factor,
        )

    except Exception as e:
        logger.error(f"Failed to initialize MinIO: {e}")
        raise

    async def _inner(input: LookAtHumanInput) -> LookAtHumanOutput:
        await wait_for_ack(input.action_uuid)
        frame_data = await capture_image(config.base_url)

        object_key = f"captures/human_{input.action_uuid}.jpg"
        try:
            result = minio.put_object(
                config.internal_minio.bucket,
                object_key,
                io.BytesIO(frame_data),
                length=len(frame_data),
                content_type="image/jpeg",
            )
            url = f"http://{config.internal_minio.base_url}/{config.internal_minio.bucket}/{result.object_name}"
            return LookAtHumanOutput(image_url_or_path=url)
        except Exception as e:
            logger.error(f"Failed to upload image to MinIO: {e}")
            raise

    yield FunctionInfo.create(
        single_fn=_inner,
        single_output_schema=LookAtHumanOutput,
        input_schema=LookAtHumanInput,
        description="Look at the user and capture an image of them."
        " This function should be called when the user is detected.",
    )
