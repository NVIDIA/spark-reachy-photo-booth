# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from collections.abc import Callable, Coroutine
from typing import Any

import statesman
import torch
from constants import UNDEFINED_INSTANCE_ID


class UserTrackerStateMachine(statesman.StateMachine):
    """User tracking state machine."""

    class States(statesman.StateEnum):
        """The states of the user tracker state machine."""

        off = statesman.InitialState("OFF")
        on = "ON"
        appeared = "USER_APPEARED"
        disappeared = "USER_DISAPPEARED"

    class Config:
        guard_with = statesman.Guard.warning
        arbitrary_types_allowed = True

    user_id: int = UNDEFINED_INSTANCE_ID
    _last_detection_time: float | None = None
    patience: float
    vicinity_threshold: float
    distance_threshold: float
    callback: (
        (Callable[[statesman.State, str, dict[str, Any]], Coroutine[Any, Any, None]])
        | None
    ) = None
    """Callback to call when the state changes.

    Args:
        state (States): The state that the user tracker is in.
        event_name (str): The name of the event that occurred.
        kwargs (dict[str, Any]): The keyword arguments to pass to the callback.
    """

    def get_elapsed_time(self) -> float | None:
        """Get the elapsed milliseconds since the last detection."""
        if self._last_detection_time is None:
            return None
        return float(time.time() * 1000 - self._last_detection_time)

    async def safe_trigger_event(self, event_name, *args, **kwargs):
        """Wrapper for trigger_event with exception handling"""
        try:
            await self.trigger_event(event_name, *args, **kwargs)
        except RuntimeError as e:
            if self._config.guard_with == statesman.Guard.warning:
                print(f"Invalid transition: {e}")
            elif self._config.guard_with == statesman.Guard.silence:
                pass
            else:
                raise

    async def handle_detection(
        self, bbox: torch.Tensor, value: float, user_id: int
    ) -> None:
        """Handle the detection of a user.

        Args:
            bbox (torch.Tensor): The bounding box of the user.
            value (float): The value of the user.
            user_id (int): The ID of the user.
        """
        if not self.is_far_away(bbox, value, user_id):
            self._last_detection_time = time.time() * 1000
        elif self._last_detection_time is None:
            # Initialize timer on first detection, even if far away
            self._last_detection_time = time.time() * 1000

        elapsed_time = time.time() * 1000 - self._last_detection_time
        if self.is_close_proximity(bbox, value, user_id):
            await self.safe_trigger_event(
                "user_detected",
                bbox,
                value,
                user_id,
                elapsed_time,
            )
        elif (
            self.is_far_away(bbox, value, user_id)
            and self.state == self.States.appeared
            and elapsed_time > self.patience
        ):
            await self.safe_trigger_event(
                "user_disappeared",
                bbox,
                value,
                user_id,
                elapsed_time,
            )

    async def handle_no_detection(self) -> None:
        """Handle the no detection of a user."""
        if self._last_detection_time is None or self.state != self.States.appeared:
            return
        elapsed_time = time.time() * 1000 - self._last_detection_time
        if elapsed_time > self.patience:
            await self.safe_trigger_event("user_disappeared", elapsed_time=elapsed_time)

    def is_close_proximity(
        self, bbox: torch.Tensor, value: float, user_id: int
    ) -> bool:
        """Check if the user is in close proximity.

        Args:
            bbox (torch.Tensor): The bounding box of the user.
            value (float): The value of the user.
            user_id (int): The ID of the user.

        Returns:
            bool: True if the user is in close proximity, False otherwise.
        """

        return value > self.vicinity_threshold

    def is_far_away(self, bbox: torch.Tensor, value: float, user_id: int) -> bool:
        """Check if the user is far away.

        Args:
            bbox (torch.Tensor): The bounding box of the user.
            value (float): The value of the user.
            user_id (int): The ID of the user.

        Returns:
            bool: True if the user is far away, False otherwise.
        """

        return value < self.distance_threshold

    @statesman.event([States.on, States.disappeared], States.appeared)  # type: ignore (missing type)
    async def user_detected(
        self, bbox: torch.Tensor, value: float, user_id: int, elapsed_time: float
    ) -> None:
        """
        User detected event.
        Note: user_id is updated to the user_id of the detected user. This is
        used to track the user across frames.
        """

        self.user_id = user_id
        if self.callback is not None and self.state is not None:
            await self.callback(
                self.state,
                "user_detected",
                {
                    "bbox": bbox,
                    "value": value,
                    "user_id": user_id,
                    "elapsed_time": elapsed_time,
                },
            )

    @statesman.event(States.appeared, States.disappeared)  # type: ignore (missing type)
    async def user_disappeared(
        self,
        bbox: torch.Tensor | None = None,
        value: float = 0.0,
        user_id: int = 0,
        elapsed_time: float = 0.0,
    ) -> None:
        """
        User disappeared event.
        Note: user_id is updated to UNDEFINED_INSTANCE_ID.
        """

        self.user_id = UNDEFINED_INSTANCE_ID
        if self.callback is not None and self.state is not None:
            await self.callback(
                self.state,
                "user_disappeared",
                {
                    "bbox": bbox,
                    "value": value,
                    "user_id": user_id,
                    "elapsed_time": elapsed_time,
                },
            )

    @statesman.event(States.off, States.on)  # type: ignore
    async def tracker_on(self) -> None:
        """Handle transition to on state"""
        if self.callback is not None and self.state is not None:
            await self.callback(self.state, "command", {})

    @statesman.event([States.on, States.appeared, States.disappeared], States.off)  # type: ignore
    async def tracker_off(self) -> None:
        """Handle transition to off state"""
        if self.callback is not None and self.state is not None:
            await self.callback(self.state, "command", {})
