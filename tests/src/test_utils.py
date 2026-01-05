# Copyright 2025 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Test utilities for integration testing of Spark & Reachy Photo Booth services via Kafka.
"""

import asyncio
import time
from collections.abc import Callable
from typing import Any

from workmesh.config import ConsumerConfig
from workmesh.service import Consumer, Topic


class ConditionTimeoutError(TimeoutError):
    """Timeout waiting for condition to be met."""

    pass


class CountTimeoutError(TimeoutError):
    """Timeout waiting for minimum message count."""

    pass


class MessageListener:
    """Context manager for listening to Kafka topics in tests.

    Usage:
        async with MessageListener(topic, consumer_config) as listener:
            # Send your test messages
            await producer.publish(...)

            # Wait for responses
            await listener.wait_for(min_count=1, timeout=30.0)

            # Access collected messages
            assert len(listener.messages) > 0

            # Clear messages for next phase
            listener.clear()

            # Wait for next set of messages
            await listener.wait_for(min_count=1, timeout=30.0)
    """

    def __init__(self, topic: Topic, consumer_config: ConsumerConfig):
        """Initialize message listener.

        Args:
            topic: Topic to subscribe to
            consumer_config: Consumer configuration
        """
        self.topic = topic
        self.messages: list[Any] = []
        self._consumer = Consumer(config=consumer_config)
        self._consumer.subscribe(topic, self._on_message)
        self._consumer_task: asyncio.Task | None = None

    async def _on_message(self, msg: Any) -> None:
        """Callback for received messages."""
        self.messages.append(msg)

    async def _run_consumer(self) -> None:
        """Run consumer loop in background."""
        try:
            while await self._consumer.consume() is not None:
                pass
        except asyncio.CancelledError:
            pass

    async def wait_for(
        self,
        min_count: int | None = None,
        timeout: float = 30.0,
        condition: Callable[[list[Any]], bool] | None = None,
    ) -> None:
        """Wait for messages to arrive or a condition to be met.

        Args:
            min_count: Minimum number of messages to wait for
                      (optional if condition provided)
            timeout: Maximum time to wait (seconds)
            condition: Optional callable that takes messages list
                      and returns True when satisfied

        Raises:
            TimeoutError: If min_count not reached or condition not met
                         within timeout
            ValueError: If neither min_count nor condition is provided
        """
        if min_count is None and condition is None:
            raise ValueError("Either min_count or condition must be provided")

        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check condition first if provided
            if condition is not None and condition(self.messages):
                return
            # Otherwise check min_count
            if min_count is not None and len(self.messages) >= min_count:
                return
            await asyncio.sleep(0.1)

        # Provide better error message based on what was checked
        if condition is not None:
            raise ConditionTimeoutError(
                f"Condition not met within {timeout}s "
                f"(received {self.messages}) messages on topic '{self.topic.name}')"
            )
        else:
            raise CountTimeoutError(
                f"Only received {len(self.messages)}/{min_count} messages "
                f"on topic '{self.topic.name}' within {timeout}s"
            )

    def clear(self) -> None:
        """Clear all collected messages.

        Useful for resetting state between different phases of a test.
        """
        self.messages.clear()

    async def __aenter__(self):
        """Start consuming when entering context."""
        self._consumer_task = asyncio.create_task(self._run_consumer())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop consuming and cleanup when exiting context."""
        if self._consumer_task:
            self._consumer_task.cancel()
            await asyncio.wait_for(self._consumer_task, timeout=10.0)
        await self._consumer.close()
