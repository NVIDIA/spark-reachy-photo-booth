# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-time low pass filters for signal processing in computer vision applications.

This module provides various low pass filter implementations optimized for real-time
processing of tracking data, bounding boxes, and coordinate streams.
"""

from abc import ABC, abstractmethod
from enum import Enum

import cv2
import numpy as np
import torch
from models.detector.base import DetectionResult
from scipy.signal import butter, lfilter, lfilter_zi


class FilterMethod(Enum):
    EMA = "ema"
    BUTTERWORTH = "butterworth"
    ADAPTIVE = "adaptive"
    KALMAN = "kalman"
    NONE = "none"


class BaseLowPassFilter(ABC):
    """Base class for all low pass filters."""

    def __init__(self, name: str = "BaseLowPassFilter"):
        self.name = name
        self.initialized = False

    @abstractmethod
    def filter(self, value: float) -> float:
        """Apply the filter to a new value.

        Args:
            value(float): Input value (scalar)

        Returns:
            float: Filtered value

        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the filter state."""
        pass


class ExponentialMovingAverageFilter(BaseLowPassFilter):
    """Exponential Moving Average (EMA) low pass filter for real-time processing.

    This is the most commonly used filter for real-time applications due to its
    simplicity and effectiveness. It uses the formula:
    y[n] = α * x[n] + (1 - α) * y[n-1]

    Where α (alpha) controls the smoothing factor:
    - α close to 1.0: Less smoothing, more responsive
    - α close to 0.0: More smoothing, less responsive
    """

    def __init__(self, alpha: float = 0.3, name: str = "EMA"):
        """Initialize the EMA filter.

        Args:
            alpha (float): Smoothing factor between 0 and 1
            name (str): Name identifier for the filter

        """
        super().__init__(name)
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")

        self.alpha = alpha
        self.previous_value: float | None = None

    def filter(self, value: float) -> float:
        if not self.initialized:
            self.previous_value = value
            self.initialized = True
            return value

        assert self.previous_value is not None
        filtered_value = self.alpha * value + (1 - self.alpha) * self.previous_value
        self.previous_value = filtered_value

        return filtered_value

    def reset(self) -> None:
        self.initialized = False
        self.previous_value = None


class ButterworthFilter(BaseLowPassFilter):
    """Butterworth low pass filter for real-time processing.

    This filter provides better frequency domain characteristics than EMA
    but requires more computation. It's implemented using scipy's lfilter
    for real-time processing.
    """

    def __init__(
        self,
        cutoff_freq: float,
        sampling_rate: float,
        order: int = 2,
        name: str = "Butterworth",
    ):
        """Initialize the Butterworth filter.

        Args:
            cutoff_freq (float): Cutoff frequency in Hz
            sampling_rate (float): Sampling rate in Hz
            order (int): Filter order (higher = steeper rolloff)
            name (str): Name identifier for the filter

        """
        super().__init__(name)

        if cutoff_freq >= sampling_rate / 2:
            raise ValueError(
                f"Cutoff frequency ({cutoff_freq}) must be less than "
                "Nyquist frequency ({sampling_rate / 2})"
            )

        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.order = order

        # Design the filter
        nyquist = sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        self.b, self.a = butter(order, normalized_cutoff, btype="low", analog=False)  # type: ignore

        # Initialize filter state
        self.zi: np.ndarray | None = None

    def filter(self, value: float) -> float:
        # Convert scalar to array for processing
        is_scalar = np.isscalar(value)
        value_array = np.array([value]) if is_scalar else np.asarray(value)

        # Initialize filter state on first call
        if not self.initialized:
            # Initialize the filter delay line
            if value_array.ndim == 1:
                self.zi = lfilter_zi(self.b, self.a) * value_array[0]
            else:
                # For multi-dimensional arrays, initialize for each dimension
                self.zi = np.array(
                    [
                        lfilter_zi(self.b, self.a) * value_array[0, i]
                        for i in range(value_array.shape[1])
                    ]
                ).T
            self.initialized = True

        assert self.zi is not None
        # Apply filter
        if value_array.ndim == 1:
            filtered_value, self.zi = lfilter(
                self.b, self.a, [value_array[0]], zi=self.zi
            )
            result = filtered_value[0]
        else:
            # Process each column independently
            result = np.zeros_like(value_array)
            for i in range(value_array.shape[1]):
                filtered_col, zi_col = lfilter(
                    self.b, self.a, [value_array[0, i]], zi=self.zi[:, i]
                )
                result[0, i] = filtered_col[0]
                self.zi[:, i] = zi_col
            result = result[0]  # Remove the extra dimension we added

        return float(result) if is_scalar else result

    def reset(self) -> None:
        self.initialized = False
        self.zi = None


class AdaptiveFilter(BaseLowPassFilter):
    """Adaptive low pass filter that adjusts its smoothing based on signal.

    This filter uses a higher alpha (less smoothing) when the signal is changing rapidly
    and a lower alpha (more smoothing) when the signal is stable.
    """

    def __init__(
        self,
        min_alpha: float = 0.1,
        max_alpha: float = 0.7,
        sensitivity: float = 1.0,
        name: str = "Adaptive",
    ):
        """Initialize the adaptive filter.

        Args:
            min_alpha (float): Minimum smoothing factor (more smoothing)
            max_alpha (float): Maximum smoothing factor (less smoothing)
            sensitivity (float): Sensitivity to signal changes
            name (str): Name identifier for the filter

        """
        super().__init__(name)

        if not (0 <= min_alpha <= max_alpha <= 1):
            raise ValueError(
                "Alpha values must satisfy: 0 <= min_alpha <= max_alpha <= 1"
            )

        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.sensitivity = sensitivity
        self.previous_value: float | None = None

    def filter(self, value: float) -> float:
        if not self.initialized:
            self.previous_value = value
            self.initialized = True
            return value

        assert self.previous_value is not None
        change = abs(value - self.previous_value)

        # Adapt alpha based on rate of change
        normalized_change = min(change * self.sensitivity, 1.0)
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * normalized_change

        # Apply EMA with adaptive alpha
        filtered_value = alpha * value + (1 - alpha) * self.previous_value
        self.previous_value = filtered_value

        return filtered_value

    def reset(self) -> None:
        self.initialized = False
        self.previous_value = None


class KalmanFilter(BaseLowPassFilter):
    """Kalman filter for real-time processing.

    This filter uses the Kalman filter algorithm to estimate the state of a system.
    """

    def __init__(self, state_dim: int, measurement_dim: int, name: str = "Kalman"):
        """Initialize the Kalman filter.

        Args:
            state_dim (int): State dimension
            measurement_dim (int): Measurement dimension
            name (str): Name identifier for the filter

        """
        super().__init__(name)
        self.kalman = cv2.KalmanFilter(state_dim, measurement_dim)
        self.kalman.measurementMatrix = np.eye(measurement_dim, state_dim)
        self.kalman.transitionMatrix = np.eye(state_dim)
        self.kalman.processNoiseCov = np.eye(state_dim) * 0.01
        self.kalman.measurementNoiseCov = np.eye(measurement_dim) * 0.1
        self.kalman.errorCovPost = np.eye(state_dim) * 0.1
        self.kalman.statePost = np.zeros(state_dim)
        self.kalman.statePre = np.zeros(state_dim)
        self.initialized = False

    def filter(self, value: float) -> float:
        if not self.initialized:
            self.initialized = True
            return value

        if self.kalman is None:
            raise ValueError("Kalman filter not initialized")

        self.kalman.predict()
        self.kalman.correct(np.array([value]))
        return float(self.kalman.statePost[0])

    def reset(self) -> None:
        self.initialized = False
        self.kalman = None


def create_filter(filter_type: str, **kwargs) -> BaseLowPassFilter:
    """Factory function to create filters by type.

    Args:
        filter_type: Type of filter ("ema", "butterworth", "adaptive", "kalman")
        **kwargs: Filter-specific parameters

    Returns:
        BaseLowPassFilter: Configured filter instance

    Examples:
        >>> ema_filter = create_filter("ema", alpha=0.3)
        >>> butter_filter = create_filter("butterworth", sampling_rate=30.0)
        >>> adaptive_filter = create_filter("adaptive", min_alpha=0.1, max_alpha=0.7)

    """
    filter_type = filter_type.lower()

    if filter_type == "ema":
        return ExponentialMovingAverageFilter(**kwargs)
    elif filter_type == "butterworth":
        return ButterworthFilter(**kwargs)
    elif filter_type == "adaptive":
        return AdaptiveFilter(**kwargs)
    elif filter_type == "kalman":
        return KalmanFilter(**kwargs)
    else:
        available_types = ["ema", "butterworth", "adaptive"]
        raise ValueError(
            f"Unknown filter type '{filter_type}'. Available types: {available_types}"
        )


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format.

    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]

    Returns:
        float: IoU value between 0 and 1
    """
    # Intersection coordinates
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())

    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union area
    box1_area = (box1[2] - box1[0]).item() * (box1[3] - box1[1]).item()
    box2_area = (box2[2] - box2[0]).item() * (box2[3] - box2[1]).item()
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0.0


def compute_box_center_distance(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Compute Euclidean distance between box centers.

    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]

    Returns:
        float: Distance between centers
    """
    center1_x = (box1[0] + box1[2]).item() / 2
    center1_y = (box1[1] + box1[3]).item() / 2
    center2_x = (box2[0] + box2[2]).item() / 2
    center2_y = (box2[1] + box2[3]).item() / 2

    return np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)


class TrackFilters:
    """Container for all filters associated with a single tracked object."""

    def __init__(self, filter_type: str, filter_kwargs: dict):
        """Initialize filters for a tracked object.

        Args:
            filter_type: Type of filter to use
            filter_kwargs: Parameters for filter creation
        """
        self.filter_type = filter_type
        self.filter_kwargs = filter_kwargs

        # Create separate filters for each component
        self.box_filters = [
            create_filter(filter_type, **filter_kwargs) for _ in range(4)
        ]  # x1, y1, x2, y2
        self.score_filter = create_filter(filter_type, **filter_kwargs)
        self.keypoint_filters: list[
            list[BaseLowPassFilter]
        ] = []  # Will be initialized on first use

    def filter_box(self, box: torch.Tensor) -> torch.Tensor:
        """Filter a bounding box.

        Args:
            box: Box tensor [x1, y1, x2, y2]

        Returns:
            Filtered box tensor
        """
        filtered = torch.zeros_like(box)
        for i in range(4):
            filtered[i] = self.box_filters[i].filter(box[i].item())
        return filtered

    def filter_score(self, score: torch.Tensor) -> torch.Tensor:
        """Filter a confidence score.

        Args:
            score: Score tensor (scalar)

        Returns:
            Filtered score tensor
        """
        return torch.tensor(self.score_filter.filter(score.item()))

    def filter_keypoints(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Filter keypoints.

        Args:
            keypoints: Keypoint tensor [K, 3] where each row is (x, y, confidence)

        Returns:
            Filtered keypoint tensor
        """
        K = keypoints.shape[0]

        # Initialize keypoint filters if needed
        if len(self.keypoint_filters) == 0:
            for _ in range(K):
                # 3 filters per keypoint: x, y, confidence
                self.keypoint_filters.append(
                    [
                        create_filter(self.filter_type, **self.filter_kwargs)
                        for _ in range(3)
                    ]
                )

        filtered = torch.zeros_like(keypoints)
        for k in range(K):
            for coord in range(3):  # x, y, confidence
                filtered[k, coord] = self.keypoint_filters[k][coord].filter(
                    keypoints[k, coord].item()
                )
        return filtered

    def reset(self):
        """Reset all filters."""
        for f in self.box_filters:
            f.reset()
        self.score_filter.reset()
        for kp_filters in self.keypoint_filters:
            for f in kp_filters:
                f.reset()


class DetectionFilter:
    """Filters detections by matching them across frames and applying smoothing.

    This class handles the full detection filtering pipeline:
    1. Match new detections to previous ones
    2. Apply filters only to matched detections
    3. Pass through unmatched detections without filtering
    """

    def __init__(
        self,
        filter_type: str = "ema",
        filter_kwargs: dict | None = None,
        iou_threshold: float = 0.3,
        max_center_distance: float | None = None,
        max_track_age: int = 30,
    ):
        """Initialize the detection filter.

        Args:
            filter_type: Type of filter to use
                ("ema", "butterworth", "adaptive", "kalman")
            filter_kwargs: Parameters for filter creation
                (e.g., {"alpha": 0.3} for EMA)
            iou_threshold: Minimum IoU for matching detections
                (if None, uses center distance)
            max_center_distance: Maximum center distance for matching
                (if iou_threshold is None)
            max_track_age: Maximum frames to keep a track without
                updates before removing it
        """
        self.filter_type = filter_type
        self.filter_kwargs = filter_kwargs or {}
        self.iou_threshold = iou_threshold
        self.max_center_distance = max_center_distance
        self.max_track_age = max_track_age

        # Store previous detections and their filters
        self.previous_boxes: list[torch.Tensor] = []
        self.previous_scores: list[torch.Tensor] = []
        self.previous_keypoints: list[torch.Tensor] = []
        self.track_filters: list[TrackFilters] = []
        self.track_ages: list[int] = []  # Frames since last update

    def match_detections(
        self,
        current_boxes: torch.Tensor,
    ) -> list[tuple[int, int]]:
        """Match current detections to previous ones.

        Args:
            current_boxes: Current detection boxes [N, 4]

        Returns:
            List of (current_idx, previous_idx) tuples for matched detections
        """
        if len(self.previous_boxes) == 0:
            return []

        N_curr = current_boxes.shape[0]
        N_prev = len(self.previous_boxes)

        # Compute cost matrix
        cost_matrix = np.zeros((N_curr, N_prev))

        for i in range(N_curr):
            for j in range(N_prev):
                if self.iou_threshold is not None:
                    # Use IoU for matching
                    iou = compute_iou(current_boxes[i], self.previous_boxes[j])
                    # Negative because we want to maximize IoU
                    cost_matrix[i, j] = -iou
                else:
                    # Use center distance for matching
                    dist = compute_box_center_distance(
                        current_boxes[i], self.previous_boxes[j]
                    )
                    cost_matrix[i, j] = dist

        # Simple greedy matching
        # (can be replaced with Hungarian algorithm for better results)
        matches = []
        matched_curr = set()
        matched_prev = set()

        # Sort all possible matches by cost
        all_matches = []
        for i in range(N_curr):
            for j in range(N_prev):
                all_matches.append((cost_matrix[i, j], i, j))
        all_matches.sort()

        # Greedily select best matches
        for cost, i, j in all_matches:
            if i in matched_curr or j in matched_prev:
                continue

            # Check if match is good enough
            if self.iou_threshold is not None:
                iou = -cost  # Convert back from negative
                if iou < self.iou_threshold:
                    continue
            else:
                if (
                    self.max_center_distance is not None
                    and cost > self.max_center_distance
                ):
                    continue

            matches.append((i, j))
            matched_curr.add(i)
            matched_prev.add(j)

        return matches

    def filter(self, detection_result: DetectionResult) -> DetectionResult:
        """Filter a detection result by matching and smoothing.

        Args:
            detection_result: DetectionResult object (single frame)

        Returns:
            Filtered DetectionResult with the same shape
        """

        boxes = detection_result.boxes  # [N, 4]
        scores = detection_result.scores  # [N]
        keypoints = detection_result.keypoints  # [N, K, 3]
        instance_ids = detection_result.instance_ids  # [N]

        N = boxes.shape[0]

        # If no detections, age all existing tracks
        if N == 0:
            new_previous_boxes = []
            new_previous_scores = []
            new_previous_keypoints = []
            new_track_filters = []
            new_track_ages = []

            for prev_idx in range(len(self.previous_boxes)):
                self.track_ages[prev_idx] += 1
                if self.track_ages[prev_idx] < self.max_track_age:
                    new_previous_boxes.append(self.previous_boxes[prev_idx])
                    new_previous_scores.append(self.previous_scores[prev_idx])
                    new_previous_keypoints.append(self.previous_keypoints[prev_idx])
                    new_track_filters.append(self.track_filters[prev_idx])
                    new_track_ages.append(self.track_ages[prev_idx])

            # Update state
            self.previous_boxes = new_previous_boxes
            self.previous_scores = new_previous_scores
            self.previous_keypoints = new_previous_keypoints
            self.track_filters = new_track_filters
            self.track_ages = new_track_ages

            return detection_result

        # Match current detections to previous ones
        matches = self.match_detections(boxes)

        # Create output tensors
        filtered_boxes = boxes.clone()
        filtered_scores = scores.clone()
        filtered_keypoints = keypoints.clone()

        # Track which previous tracks were matched
        matched_prev_indices = set()

        # Apply filters to matched detections
        for curr_idx, prev_idx in matches:
            matched_prev_indices.add(prev_idx)

            # Apply filters
            filtered_boxes[curr_idx] = self.track_filters[prev_idx].filter_box(
                boxes[curr_idx]
            )
            filtered_scores[curr_idx] = self.track_filters[prev_idx].filter_score(
                scores[curr_idx]
            )
            filtered_keypoints[curr_idx] = self.track_filters[
                prev_idx
            ].filter_keypoints(keypoints[curr_idx])

            # Reset track age
            self.track_ages[prev_idx] = 0

        # Update previous detections and filters for next frame
        new_previous_boxes = []
        new_previous_scores = []
        new_previous_keypoints = []
        new_track_filters = []
        new_track_ages = []

        # Keep matched tracks
        for curr_idx, prev_idx in matches:
            new_previous_boxes.append(filtered_boxes[curr_idx].clone())
            new_previous_scores.append(filtered_scores[curr_idx].clone())
            new_previous_keypoints.append(filtered_keypoints[curr_idx].clone())
            new_track_filters.append(self.track_filters[prev_idx])
            new_track_ages.append(0)

        # Keep unmatched previous tracks (age them)
        for prev_idx in range(len(self.previous_boxes)):
            if prev_idx not in matched_prev_indices:
                self.track_ages[prev_idx] += 1
                if self.track_ages[prev_idx] < self.max_track_age:
                    new_previous_boxes.append(self.previous_boxes[prev_idx])
                    new_previous_scores.append(self.previous_scores[prev_idx])
                    new_previous_keypoints.append(self.previous_keypoints[prev_idx])
                    new_track_filters.append(self.track_filters[prev_idx])
                    new_track_ages.append(self.track_ages[prev_idx])

        # Add new unmatched current detections (not filtered)
        matched_curr_indices = {curr_idx for curr_idx, _ in matches}
        for curr_idx in range(N):
            if curr_idx not in matched_curr_indices:
                # New detection - add to tracks without filtering
                new_previous_boxes.append(boxes[curr_idx].clone())
                new_previous_scores.append(scores[curr_idx].clone())
                new_previous_keypoints.append(keypoints[curr_idx].clone())
                new_track_filters.append(
                    TrackFilters(self.filter_type, self.filter_kwargs)
                )
                new_track_ages.append(0)

        # Update state
        self.previous_boxes = new_previous_boxes
        self.previous_scores = new_previous_scores
        self.previous_keypoints = new_previous_keypoints
        self.track_filters = new_track_filters
        self.track_ages = new_track_ages

        # Return filtered result
        return DetectionResult(
            boxes=filtered_boxes,
            scores=filtered_scores,
            keypoints=filtered_keypoints,
            instance_ids=instance_ids,
        )

    def reset(self):
        """Reset all state."""
        self.previous_boxes = []
        self.previous_scores = []
        self.previous_keypoints = []
        self.track_filters = []
        self.track_ages = []
