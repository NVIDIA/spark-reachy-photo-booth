# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import re

import aiohttp
import regex
import urllib3
from google.protobuf.message import Message
from minio import Minio
from urllib3.util import Retry, Timeout
from workmesh.messages import ToolStatus, UserUtterance, UserUtteranceStatus
from workmesh.service import Consumer

from photo_booth_agent.constant import SERVICE_NAME
from workmesh import (
    Topic,
    routed_user_utterance_topic,
    tool_status_topic,
)

logger = logging.getLogger(SERVICE_NAME)


def initialize_minio(
    minio_base_url: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    policy: str,
    secure: bool = False,
    create_bucket: bool = True,
    timeout: float = 5.0,
    num_retries: int = 3,
    backoff_factor: float = 0.2,
) -> Minio:
    """
    Initialize the MinIO client.

    Args:
        minio_base_url (str): The base URL of the MinIO service.
        minio_access_key (str): The access key of the MinIO service.
        minio_secret_key (str): The secret key of the MinIO service.
        minio_bucket (str): The name of the MinIO bucket.
        policy (str): The policy to set on the MinIO bucket.
        secure (bool): Whether to use HTTPS.
        create_bucket (bool): Whether to create the bucket if it doesn't exist.
        timeout (float): Connection and read timeout in seconds.
        num_retries (int): Number of retries for failed requests.
        backoff_factor (float): Backoff factor for retry delay.

    Returns:
        Minio: The initialized MinIO client.
    """
    http_client = urllib3.PoolManager(
        timeout=Timeout(connect=timeout, read=timeout),
        retries=Retry(total=num_retries, backoff_factor=backoff_factor),
    )
    minio = Minio(
        minio_base_url,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=secure,
        http_client=http_client,
    )
    if not create_bucket:
        return minio
    bucket_exists = minio.bucket_exists(minio_bucket)
    if not bucket_exists:
        minio.make_bucket(minio_bucket)
        logger.info(f"Created bucket {minio_bucket}")
    else:
        logger.info(f"Bucket {minio_bucket} already exists")
    try:
        minio.set_bucket_policy(minio_bucket, policy)
    except Exception as e:
        logger.error(f"Failed to set policy on bucket {minio_bucket}: {e}")
        raise
    return minio


def clean_reasoning(llm_output: str) -> str:
    """
    Clean the reasoning output from the LLM.

    Note: reasoning output is not always well-formed, so we need to remove the tags for
    stray tags
    """
    llm_output = re.sub(
        r"<think>.*?</think>",
        "",
        llm_output,
        flags=re.DOTALL,
    )

    json_object = regex.search(
        r'\{(?:[^{}"\\]+|"(?:[^"\\]|\\.)*"|(?R))*\}',
        llm_output,
    )
    if json_object:
        llm_output = json_object.group(0)
    else:
        logger.warning("No JSON object found in the LLM output")
    return llm_output.replace("<think>", "").replace("</think>", "").strip()


async def wait_for_event(
    *,
    action_uuid: str,
    timeout: int = 60,
    topic: Topic,
    expected_type: type[Message],
    expected_status: int | None = None,
) -> None:
    """
    Wait for a message with a matching action_uuid (and optional status)
    to be received on a given topic.

    Args:
        action_uuid (str): The action UUID to match.
        timeout (int): Max seconds to wait.
        topic (Topic): Topic to consume from.
        expected_type: Type to check for the message.
        expected_status (optional): Status value to match.
    """

    async def _wait_for_event():
        waiting_message = f"Waiting for {expected_type.__name__} (uuid={action_uuid})"
        if expected_status is not None:
            waiting_message += f" with status {expected_status}"
        logger.info(waiting_message)

        consumer = Consumer()
        try:
            consumer.subscribe(topic)  # pyright: ignore[reportArgumentType]
            while True:
                result = await consumer.consume()
                if result is None:
                    continue
                message, recv_topic = result
                logger.debug(f"Received message from {recv_topic}")
                if recv_topic != topic:
                    logger.debug(f"Message from wrong topic: {recv_topic}")
                    continue
                if not isinstance(message, expected_type):
                    logger.debug(
                        f"Failed to cast message to {expected_type.__name__}: {type(message).__name__}"  # noqa: E501
                    )
                    continue
                if getattr(message, "action_uuid", None) != action_uuid:
                    logger.debug(
                        f"Message from wrong action UUID: "
                        f"{getattr(message, 'action_uuid', None)}, "
                        f"expected: {action_uuid}"
                    )
                    continue
                if (
                    expected_status is not None
                    and getattr(message, "status", None) != expected_status
                ):
                    logger.debug(
                        f"Message from wrong status: "
                        f"{getattr(message, 'status', None)}, "
                        f"expected: {expected_status}"
                    )
                    continue

                break
        finally:
            await consumer.close()

    timeout_task = asyncio.create_task(_wait_for_event())
    try:
        await asyncio.wait_for(timeout_task, timeout)
    except TimeoutError as e:
        raise TimeoutError(
            f"Timeout waiting for {expected_type.__name__} (uuid={action_uuid}) "
            f"with status {expected_status} after {timeout} seconds"
        ) from e


async def wait_for_ack(action_uuid: str, timeout: int = 60) -> None:
    """
    Wait for the tool to be processed by the interaction manager.
    """
    await wait_for_event(
        action_uuid=action_uuid,
        timeout=timeout,
        topic=tool_status_topic,
        expected_type=ToolStatus,
        expected_status=ToolStatus.TOOL_CALL_PROCESSED,
    )
    logger.info(f"Tool '{action_uuid}' processed.")


async def wait_for_user_utterance_started(action_uuid: str, timeout: int = 60) -> None:
    """
    Wait for a user utterance to be started.
    """
    await wait_for_event(
        action_uuid=action_uuid,
        timeout=timeout,
        topic=routed_user_utterance_topic,
        expected_type=UserUtterance,
        expected_status=UserUtteranceStatus.USER_UTTERANCE_STARTED,
    )
    logger.info(f"User utterance '{action_uuid}' started.")


async def capture_image(base_url: str) -> bytes:
    """
    Capture an image from the camera service.

    Args:
        base_url (str): The base URL of the camera service.
    """

    async with (
        aiohttp.ClientSession() as session,
        session.get(
            f"{base_url}/capture",
        ) as response,
    ):
        response.raise_for_status()
        return await response.content.read()
