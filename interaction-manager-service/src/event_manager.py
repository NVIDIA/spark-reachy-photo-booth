# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections import defaultdict

from workmesh.messages import ClipStatus


class EventManager:
    """Event manager that waits for clip status events."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

        # Store events by action_uuid and status
        self._events: dict[str, dict[ClipStatus.Status, asyncio.Event]] = defaultdict(
            lambda: defaultdict(asyncio.Event)
        )

        self.logger.info("EventManager initialized")

    def stop(self) -> None:
        """Stop the event manager."""

        self._events.clear()
        self.logger.info("EventManager stopped.")

    async def add_status_event(self, message: ClipStatus) -> None:
        """Add new clip status event."""

        action_uuid = message.action_uuid
        self.logger.debug(f"Received clip status event: {action_uuid} - {ClipStatus.Status.Name(message.status)}")  # noqa: E501 # fmt: skip

        # Get the event for the action_uuid and status
        event = self._events[action_uuid][message.status]

        # Set the event to wake up any waiting coroutines
        event.set()

    async def wait_for_event(self, action_uuid: str, status: ClipStatus.Status) -> None:
        """
        Wait for a specific clip status event.

        Args:
            action_uuid (str): The unique identifier for the action whose event to wait for.
            status (ClipStatus.Status): The status of the clip event to wait for.

        This coroutine will block until the specified event is set, indicating that
        the desired clip status has been reached for the given action_uuid.
        """  # noqa: E501

        # Obtain the event for the action_uuid and status
        event = self._events[action_uuid][status]

        # Wait for the event to be set
        status_name = ClipStatus.Status.Name(status)
        self.logger.debug(f"Waiting for event: {action_uuid} - {status_name}")
        await event.wait()
        self.logger.debug(f"End waiting for event: {action_uuid} - {status_name}")

    async def wait_for_clip_finished(self, action_uuid: str) -> None:
        """
        Waits for the clip associated with the given action_uuid to reach either
        the FINISHED or ERROR status.

        This coroutine suspends execution until the clip with action_uuid receives
        a FINISHED or ERROR event. Upon receiving either event, all status events
        for that action_uuid are cleared. This function does not return a value but
        is used to await completion or failure of the clip.

        Args:
            action_uuid (str): The unique identifier of the clip to monitor.
        """

        # Wait for either FINISHED or ERROR event (whichever comes first)
        finished_task = asyncio.create_task(
            self.wait_for_event(action_uuid, ClipStatus.Status.FINISHED),
            name="wait_for_clip_finished",
        )
        error_task = asyncio.create_task(
            self.wait_for_event(action_uuid, ClipStatus.Status.ERROR),
            name="wait_for_clip_error",
        )

        done, pending = await asyncio.wait(
            {finished_task, error_task}, return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel any pending task
        for task in pending:
            task.cancel()

        if error_task in done:
            self.logger.error(f"Clip error: {action_uuid}")
        else:
            self.logger.debug(f"Clip finished: {action_uuid}")

        self.clear_clip_events(action_uuid)

    async def wait_for_clip_started(self, action_uuid: str) -> bool:
        """
        Wait for the clip with the given action_uuid to reach the STARTED status,
        or return False if an error occurs.

        This coroutine waits until the clip with the given action_uuid either
        reaches the STARTED status, indicating that clip processing has begun,
        or encounters an ERROR status. If STARTED is reached, returns True;
        if ERROR occurs first, returns False.

        Args:
            action_uuid (str): The unique identifier for the action whose
            clip to wait for.

        Returns:
            bool: True if clip started successfully, False if an error occurred.
        """
        # Wait for either STARTED or ERROR event (whichever comes first)
        started_task = asyncio.create_task(
            self.wait_for_event(action_uuid, ClipStatus.Status.STARTED),
            name="wait_for_clip_started",
        )
        error_task = asyncio.create_task(
            self.wait_for_event(action_uuid, ClipStatus.Status.ERROR),
            name="wait_for_clip_error",
        )

        done, pending = await asyncio.wait(
            {started_task, error_task}, return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel any pending task
        for task in pending:
            task.cancel()

        if error_task in done:
            self.logger.error(f"Clip error: {action_uuid}")
            return False

        self.logger.debug(f"Clip started: {action_uuid}")
        return True

    def clear_clip_events(self, action_uuid: str) -> None:
        """Clear events for a specific action_uuid."""

        self._events.pop(action_uuid, None)
        self.logger.debug(f"Cleared all events for action: {action_uuid}")
