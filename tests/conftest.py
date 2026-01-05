# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for integration tests."""

import os
from collections.abc import AsyncGenerator

import pytest
from pydantic.networks import KafkaDsn
from workmesh.config import ConsumerConfig, ProducerConfig
from workmesh.service import Producer


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--kafka-url",
        action="store",
        default=None,
        help="Kafka broker URL (e.g., kafka://localhost:19092)",
    )


@pytest.fixture(scope="session")
def kafka_broker_url(request) -> str:
    """Provide Kafka broker URL from CLI option or environment variable.

    Priority:
    1. --kafka-url command line option
    2. KAFKA_BROKER_URL environment variable
    3. Default: kafka://localhost:19092

    Returns:
        Kafka broker URL string
    """
    # Check CLI option
    url = request.config.getoption("--kafka-url")
    if url:
        return url

    # Check environment variable
    url = os.environ.get("KAFKA_BROKER_URL")
    if url:
        return url

    # Default to localhost
    return "kafka://localhost:19092"


@pytest.fixture
async def producer(kafka_broker_url: str) -> AsyncGenerator[Producer, None]:
    """Provide a workmesh Producer instance with automatic cleanup.

    Args:
        kafka_broker_url: Kafka broker URL from kafka_broker_url fixture

    Yields:
        Producer instance
    """
    config = ProducerConfig(broker_url=KafkaDsn(kafka_broker_url))
    producer = Producer(config=config)
    try:
        yield producer
    finally:
        await producer.stop()


@pytest.fixture
def consumer_config(kafka_broker_url: str) -> ConsumerConfig:
    """Provide ConsumerConfig for creating consumers.

    Args:
        kafka_broker_url: Kafka broker URL from kafka_broker_url fixture

    Returns:
        ConsumerConfig instance
    """
    import time
    import uuid

    # Generate unique consumer group for each test run
    consumer_group = f"test-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    return ConsumerConfig(
        broker_url=KafkaDsn(kafka_broker_url),
        consumer_group=consumer_group,
    )
