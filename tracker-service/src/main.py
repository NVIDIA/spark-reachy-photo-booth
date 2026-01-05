# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
import traceback
from asyncio import Queue
from collections.abc import AsyncGenerator
from typing import Any

import cv2
import numpy as np
import statesman
import torch
from configuration import FilterConfig, TrackerConfig
from constants import UNDEFINED_INSTANCE_ID
from helpers import filters, utils
from models.detector import create_detector
from models.detector.base import DetectionResult
from state import UserTrackerStateMachine
from workmesh.config import load_config
from workmesh.messages import (
    BoundingBox,
    Command,
    Frame,
    Keypoint,
    PresenceStatus,
    Robot,
    ServiceCommand,
    Skeleton,
    UserDetection,
    UserState,
)
from workmesh.messages import (
    Service as ServiceName,
)
from workmesh.service import subscribe
from workmesh.service_executor import ServiceExecutor

from workmesh import (
    Service,
    camera_frame_topic,
    service_command_topic,
    user_detection_topic,
    user_state_topic,
)

USER_ID = 0


async def main():
    config = load_config(TrackerConfig)
    await ServiceExecutor([TrackerService(config)]).run()


class TrackerService(Service):
    """Tracker service that runs the tracking task."""

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        self._queue: Queue[Frame] = Queue(maxsize=1)
        self._config: TrackerConfig = config
        self._me = getattr(Robot, self._config.robot_id.name)

        self._det_filter = self._initialize_filters(self._config.filter)
        self._model = create_detector(self._config.model_type, self._config.model)

        self._enable_tracker = False

        self.logger.info(f"TrackerService initialized with model: {self._model}")
        self.logger.info(f"TrackerService initialized with config: {self._config}")

        # Initialize the transitions-based state machine
        self._tracker = UserTrackerStateMachine(
            patience=self._config.patience,
            vicinity_threshold=self._config.vicinity_threshold,
            distance_threshold=self._config.distance_threshold,
            callback=self._on_state_change,
        )

        self._task = asyncio.create_task(self.run_tracker())

        self._fps = self._config.fps
        self._switch_patience_start_time: float | None = None

    async def _on_state_change(
        self,
        new_state: statesman.State,
        trigger: str,
        context: dict[str, Any],
    ) -> None:
        """
        Called when state machine transitions to a new state.

        Args:
            new_state (UserTrackerStateMachine.States): The new state.
            trigger (str): The trigger that caused the state change.
            context (dict[str, Any]): The context that caused the state change.
        """

        self.logger.info(
            f"State transition to {new_state} (trigger: {trigger}, context: {context})"
        )

        if (
            new_state == UserTrackerStateMachine.States.appeared
            or new_state == UserTrackerStateMachine.States.disappeared
        ):
            await self.publish(
                user_state_topic,
                UserState(
                    robot_id=self._me,
                    status=getattr(PresenceStatus, str(new_state.description)),
                    user_id=USER_ID,
                ),
            )

    @subscribe(service_command_topic)
    async def consume_service_command(self, command: ServiceCommand) -> None:
        if command.target_service != ServiceName.TRACKER:
            return
        if command.command == Command.ENABLE and not self._enable_tracker:
            self.logger.info("Tracker inference started.")
            self._enable_tracker = True
            await self._tracker.safe_trigger_event("tracker_on")
        elif command.command == Command.DISABLE and self._enable_tracker:
            self.logger.info("Tracker inference stopped.")
            self._enable_tracker = False
            # Discard the latest frame
            if not self._queue.empty():
                self._queue.get_nowait()
            await self._tracker.safe_trigger_event("tracker_off")

    @subscribe(camera_frame_topic)
    async def consume_frame(self, frame: Frame) -> None:
        """Consume frames from the camera topic."""

        if frame.robot_id != self._me:
            self.logger.debug(f"Skipping frame: {frame.index} because it is not for me")
            return

        if self._enable_tracker:
            # Always keep the latest frame - discard old one if queue is full
            if self._queue.full():
                self._queue.get_nowait()
            self._queue.put_nowait(frame)

    async def run_tracker(self) -> AsyncGenerator[UserDetection, Any]:
        """Run the tracking task.

        Note: This task is run in the background and publishes the user tracker.
        """

        last_frame_time = time.time()
        while True:
            # Wait until enough time has passed based on FPS throttling
            current_time = time.time()
            time_since_last = current_time - last_frame_time
            min_frame_interval = 1.0 / self._fps

            if time_since_last < min_frame_interval:
                await asyncio.sleep(min_frame_interval - time_since_last)

            # Now get the latest frame from queue
            frame = await self._queue.get()
            last_frame_time = time.time()

            # Skip processing if tracker is disabled
            if not self._enable_tracker:
                self.logger.info("Skipping frame. Tracker is disabled")
                continue

            try:
                detection_results = await self._process_frame(frame)
                if detection_results is not None:
                    detection_result, marker_id = detection_results
                    await self.publish(
                        user_detection_topic,
                        UserDetection(
                            robot_id=self._me,
                            user_id=USER_ID,
                            bounding_boxes=[
                                BoundingBox(
                                    top_left_x=bbox[0].item(),
                                    top_left_y=bbox[1].item(),
                                    width=bbox[2].item() - bbox[0].item(),
                                    height=bbox[3].item() - bbox[1].item(),
                                    score=score.item(),
                                    frame_index=frame.index,
                                    robot_id=self._me,
                                )
                                for bbox, score in zip(
                                    detection_result.boxes,
                                    detection_result.scores,
                                    strict=True,
                                )
                            ],
                            skeletons=[
                                Skeleton(
                                    keypoints=[
                                        Keypoint(
                                            x=keypoint[0].item(),
                                            y=keypoint[1].item(),
                                            confidence=keypoint[2].item(),
                                        )
                                        for keypoint in skeleton
                                    ]
                                )
                                for skeleton in detection_result.keypoints
                            ],
                            marker_id=marker_id,
                        ),
                    )
            except Exception as e:
                traceback.print_exc()
                self.logger.error(f"Error processing frame: {e}")
                continue

    async def _process_frame(self, frame: Frame) -> tuple[DetectionResult, int] | None:
        """
        Process a single frame and return the detection result.

        Args:
            frame (Frame): The frame to process.

        Returns:
            tuple[DetectionResult, int] | None: The detection result and the marker id.
        """

        # Run heavy computation in executor
        detection_results = await asyncio.get_running_loop().run_in_executor(
            None, self._run_detection, frame
        )

        if detection_results is None:
            return await self._handle_no_detection()
        detection_result, marker_id = detection_results
        return await self._handle_detection(detection_result, marker_id)

    def _run_detection(self, frame: Frame) -> tuple[DetectionResult, int] | None:
        """
        Run detection on frame and return the best bounding box candidate.

        Args:
            frame (Frame): The frame to run detection on.

        Returns:
            tuple[DetectionResult, int] | None: The detection result and the marker id.
        """

        # Decode & Preprocess Frame
        frame_data = np.frombuffer(frame.data, dtype=np.uint8)
        frame_data = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        frame_data = cv2.resize(frame_data, (self._config.width, self._config.height))
        frame_data = utils.from_cv2_to_torch(frame_data)

        # Run model inference
        detection_result = self._model.run(frame_data)

        detection_result = detection_result.normalize_bounding_box(
            original_width=self._config.width,
            original_height=self._config.height,
        )

        # Check if filtering removed all detections
        if len(detection_result.boxes) == 0:
            return None

        # Get sort values (with center penalty if enabled)
        sort_values = self._get_sort_values(detection_result)

        if (
            torch.all(detection_result.instance_ids == UNDEFINED_INSTANCE_ID)
            or self._tracker.state != UserTrackerStateMachine.States.appeared
        ):
            marker_id = torch.argmax(sort_values).item()
        else:
            # The user is definitely appeared
            tracker_matches_indices = (
                detection_result.instance_ids == self._tracker.user_id
            ).nonzero(as_tuple=True)[0]

            biggest_bbox_index = torch.argmax(sort_values).item()

            if self._should_switch_marker(
                detection_result, int(biggest_bbox_index), tracker_matches_indices
            ):
                marker_id = biggest_bbox_index
            else:
                marker_id = tracker_matches_indices[0].item()

        return detection_result, int(marker_id)

    def _get_sort_values(self, detection_result: DetectionResult) -> torch.Tensor:
        """Get the sort values for detection result based on config.

        Applies center penalty if enabled in configuration.

        Args:
            detection_result: The detection result

        Returns:
            torch.Tensor: The values to sort by (penalized if enabled)
        """

        base_values = getattr(detection_result, self._config.sort_key)

        if not self._config.center_penalty_enabled:
            return base_values

        # Apply center penalty
        penalty = detection_result.compute_center_penalty(
            min_penalty=self._config.center_penalty_min,
            penalty_type=self._config.center_penalty_type,
        )
        penalized_values = base_values * penalty

        # Debug logging
        if detection_result.boxes.shape[0] > 0:
            self.logger.debug(
                f"Center penalty debug - "
                f"sort_key={self._config.sort_key}, "
                f"penalty_type={self._config.center_penalty_type}, "
                f"min_penalty={self._config.center_penalty_min}"
            )
            for i in range(detection_result.boxes.shape[0]):
                box = detection_result.boxes[i]
                center_x = (box[0] + box[2]) / 2.0
                center_y = (box[1] + box[3]) / 2.0
                self.logger.debug(
                    f"  Box {i}: center=({center_x:.3f}, {center_y:.3f}), "
                    f"penalty={penalty[i]:.3f}, "
                    f"base_value={base_values[i]:.4f}, "
                    f"penalized={penalized_values[i]:.4f}"
                )

        return penalized_values

    def _should_switch_marker(
        self,
        detection_result: DetectionResult,
        biggest_bbox_index: int,
        tracker_matches: torch.Tensor,
    ) -> bool:
        """Check if the tracked bbox should be switched to the biggest bbox.

        Args:
            detection_result (DetectionResult): The detection result.
            biggest_bbox_index (int): The index of the biggest bbox.
            tracker_matches (torch.Tensor): The indices of the tracker matches.
        """
        # No tracking matches found - switch to biggest
        if tracker_matches.numel() == 0:
            self.logger.info("No tracking matches found - switching to biggest bbox")
            self._switch_patience_start_time = None
            return True

        # Biggest box is already the tracked one - keep it
        if (tracker_matches == biggest_bbox_index).any():
            self.logger.info("Biggest box is already the tracked one - keeping it")
            self._switch_patience_start_time = None
            return True

        # Get sort values (with center penalty if enabled)
        sort_values = self._get_sort_values(detection_result)

        biggest_box_is_better_than_tracked = (
            sort_values[biggest_bbox_index] - sort_values[tracker_matches[0]]
        ) > self._config.marker_switch_threshold

        difference = abs(
            sort_values[biggest_bbox_index] - sort_values[tracker_matches[0]]
        )

        if biggest_box_is_better_than_tracked:
            current_time = time.time()
            if self._switch_patience_start_time is None:
                self._switch_patience_start_time = current_time
                self.logger.info(
                    f"Biggest box is better than tracked "
                    f"(difference: {difference:.3f}, "
                    f"starting patience timer)"
                )
                return False

            elapsed_ms = (current_time - self._switch_patience_start_time) * 1000
            self.logger.info(
                f"Biggest box is better than tracked "
                f"(difference: {difference:.3f}, "
                f"elapsed: {elapsed_ms:.0f}ms/{self._config.marker_switch_patience}ms)"
            )
            # Only switch if patience threshold is reached
            return elapsed_ms >= self._config.marker_switch_patience
        else:
            # Reset timer if condition no longer holds
            if self._switch_patience_start_time is not None:
                self.logger.info(
                    f"Resetting switch patience timer (difference: {difference:.3f})"
                )
            self._switch_patience_start_time = None
            return False

    async def _handle_no_detection(self) -> None:
        """Handle case when no detection is found."""

        await self._tracker.handle_no_detection()

    async def _handle_detection(
        self, detection_result: DetectionResult, marker_id: int
    ) -> tuple[DetectionResult, int] | None:
        """Handle case when detection is found.

        Args:
            detection_result (DetectionResult): The detection result.
            Note: The bounding box is already normalized to be in the range of 0 to 1.
            marker_id (int): The marker id of the best bounding box candidate.

        Returns:
            tuple[DetectionResult, int] | None: The detection result and the marker id.
        """

        if self._det_filter is not None:
            detection_result = self._det_filter.filter(detection_result)
        value = getattr(detection_result, self._config.sort_key)[marker_id].item()
        instance_id = detection_result.instance_ids[marker_id].item()
        await self._tracker.handle_detection(
            detection_result.boxes[marker_id],
            value,
            int(instance_id),
        )
        return (
            (detection_result, marker_id)
            if self._tracker.state
            not in [
                UserTrackerStateMachine.States.disappeared,
                UserTrackerStateMachine.States.off,
            ]
            else None
        )

    def _initialize_filters(
        self, filter_config: FilterConfig
    ) -> filters.DetectionFilter | None:
        if filter_config.name is None:
            return None
        if filter_config.name == filters.FilterMethod.EMA.value:
            det_filter = filters.DetectionFilter(
                filter_config.name, filter_kwargs={"alpha": filter_config.alpha}
            )
        elif filter_config.name == filters.FilterMethod.BUTTERWORTH.value:
            det_filter = filters.DetectionFilter(
                filter_config.name,
                filter_kwargs={
                    "cutoff_freq": filter_config.cutoff_frequency,
                    "sampling_rate": filter_config.sample_rate,
                    "order": filter_config.order,
                },
            )
        elif filter_config.name == filters.FilterMethod.ADAPTIVE.value:
            det_filter = filters.DetectionFilter(
                filter_config.name,
                filter_kwargs={
                    "min_alpha": filter_config.min_alpha,
                    "max_alpha": filter_config.max_alpha,
                    "sensitivity": filter_config.sensitivity,
                },
            )
        elif filter_config.name == filters.FilterMethod.KALMAN.value:
            det_filter = filters.DetectionFilter(
                filter_config.name,
                filter_kwargs={
                    "state_dim": filter_config.state_dim,
                    "measurement_dim": filter_config.measurement_dim,
                },
            )
        else:
            raise ValueError(f"Invalid filter method: {filter_config.name}")
        return det_filter

    def _reset_filters(self) -> None:
        if self._det_filter is not None:
            self._det_filter.reset()

    async def stop(self) -> None:
        """Stop the tracker service."""
        self._task.cancel()
        if self._det_filter is not None:
            self._reset_filters()
        await super().stop()


if __name__ == "__main__":
    asyncio.run(main())
