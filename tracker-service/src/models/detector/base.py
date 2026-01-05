# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from abc import ABC, abstractmethod
from typing import Annotated

import cv2
import numpy as np
import torch
from constants import NUM_KEYPOINTS, SKELETON_CONNECTIONS
from pydantic import BaseModel, model_validator


class DetectionResult(BaseModel):
    class Config:
        arbitrary_types_allowed = True  # torch tensors are not allowed by default

    boxes: Annotated[torch.Tensor, "Shape: (N, 4) (x1, y1, x2, y2)"]
    scores: Annotated[torch.Tensor, "Shape: (N,)"]
    instance_ids: Annotated[torch.Tensor, "Shape: (N,)"]
    keypoints: Annotated[
        torch.Tensor,
        "Shape: (N, K, 3) (x, y, confidence). "
        "The first point is the nose, the rest are the other points of the skeleton. ",
    ]

    @model_validator(mode="after")
    def check_detection_result(self) -> "DetectionResult":
        assert len(self.keypoints.shape) == 3, "Keypoints must have 3 dimensions"
        N, K, _ = self.keypoints.shape
        assert K == NUM_KEYPOINTS or K == 0, (
            f"Keypoints must have {NUM_KEYPOINTS} keypoints but got {K}"
        )
        assert self.keypoints.shape[2] == 3, (
            f"Keypoints must have shape (N, K, 3) but got {self.keypoints.shape}"
        )
        assert self.boxes.shape == (N, 4), (
            f"Boxes must have shape (N, 4) but got {self.boxes.shape}"
        )
        assert self.scores.shape == (N,), (
            f"Scores must have shape (N,) but got {self.scores.shape}"
        )
        assert self.instance_ids.shape == (N,), (
            f"Instance IDs must have shape (N,) but got {self.instance_ids.shape}"
        )
        return self

    @property
    def area(self) -> torch.Tensor:
        """Calculate the area of the bounding boxes.

        Returns:
            torch.Tensor: The area of the bounding boxes. Shape: (N,)
        """
        assert torch.all(self.boxes >= -1e-3) and torch.all(self.boxes <= 1.0 + 1e-3), (
            "Bounding boxes must be normalized to [0, 1] range "
            "before computing area. "
            f"Found values outside range: "
            f"min={self.boxes.min().item():.3f}, "
            f"max={self.boxes.max().item():.3f}"
        )
        return abs(self.boxes[:, 2] - self.boxes[:, 0]) * abs(
            self.boxes[:, 3] - self.boxes[:, 1]
        )

    @property
    def score(self) -> torch.Tensor:
        """Calculate the score of the bounding boxes.

        Returns:
            torch.Tensor: The score of the bounding boxes.
        """
        return self.scores

    def compute_center_penalty(
        self,
        min_penalty: float = 0.8,
        penalty_type: str = "linear",
        max_distance: float | None = None,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """Compute penalty factor based on distance from frame center.

        Boxes at the center of the frame get no penalty (factor 1.0),
        while boxes at the edges get increasingly penalized.

        Args:
            min_penalty: Minimum penalty factor at the edge (default 0.8)
            penalty_type: Type of penalty function ("linear", "quadratic", "cubic")
            max_distance: Maximum distance from center (default sqrt(0.5))
                If None, uses the diagonal distance from center to corner
            eps: Small tolerance for floating point precision (default 1e-3)

        Returns:
            torch.Tensor: Penalty factors for each box. Shape: (N,)
        """
        if self.boxes.shape[0] == 0:
            return torch.ones(0, device=self.boxes.device)

        assert torch.all(self.boxes >= -eps) and torch.all(self.boxes <= 1.0 + eps), (
            "Bounding boxes must be normalized to [0, 1] range "
            "before computing center penalty. "
            f"Found values outside range: "
            f"min={self.boxes.min().item():.3f}, "
            f"max={self.boxes.max().item():.3f}"
        )

        # Compute box centers (normalized coordinates 0-1)
        box_centers_x = (self.boxes[:, 0] + self.boxes[:, 2]) / 2.0
        # Frame center is at (0.5, 0.5)
        frame_center_x = 0.5

        # Compute distance from center horizontally
        distance = torch.abs(box_centers_x - frame_center_x)

        # Maximum possible distance (from center to corner)
        max_distance = max_distance or 0.5

        # Normalize distance to [0, 1] range
        normalized_distance = torch.clamp(distance / max_distance, 0.0, 1.0)

        # Apply penalty function based on type
        if penalty_type == "linear":
            # Linear penalty: 1.0 at center, min_penalty at edge
            penalty = 1.0 - normalized_distance * (1.0 - min_penalty)
        elif penalty_type == "quadratic":
            # Quadratic penalty: slower near center, faster near edges
            penalty = 1.0 - (normalized_distance**2) * (1.0 - min_penalty)
        elif penalty_type == "cubic":
            # Cubic penalty: even slower near center, very fast near edges
            penalty = 1.0 - (normalized_distance**3) * (1.0 - min_penalty)
        else:
            raise ValueError(
                f"Invalid penalty_type: {penalty_type}. "
                "Must be 'linear', 'quadratic', or 'cubic'."
            )

        return penalty

    def normalize_bounding_box(
        self, original_width: int, original_height: int
    ) -> "DetectionResult":
        """Normalize the bounding boxes to the range of 0 to 1.

        Args:
            original_width (int): The original width of the image.
            original_height (int): The original height of the image.
        """
        return DetectionResult(
            boxes=self.boxes
            / torch.tensor(
                [original_width, original_height, original_width, original_height],
                device=self.boxes.device,
            ),
            scores=self.scores,
            instance_ids=self.instance_ids,
            keypoints=self.keypoints
            / torch.tensor(
                [original_width, original_height, 1.0], device=self.keypoints.device
            ),
        )


class BaseDetector(ABC):
    @abstractmethod
    def run(self, video: torch.Tensor) -> DetectionResult:
        """Run the detector on the video.

        Args:
            video (torch.Tensor): The video to run the detector on. Shape: (T, C, H, W)
            Note: T must be 1 for now.

        Returns:
            DetectionResult: the detection result object.

        """
        pass

    def visualize(
        self,
        frame: torch.Tensor,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        filename: str,
        keypoints: torch.Tensor,
        visibility_threshold: float = 1e-1,
    ) -> None:
        """Visualize the detections on the video.

        Args:
            frame (torch.Tensor): Frame to visualize the detections on.
                Shape: (3, H, W)
            boxes (torch.Tensor): Boxes to visualize. Shape: (N, 4)
            scores (torch.Tensor): Scores to visualize. Shape: (N,)
            filename (str): Filename to save the visualization to.
            keypoints (torch.Tensor): Keypoints. Shape: (N, K, 3)
            visibility_threshold (float): Keypoint visibility threshold.
        """

        init_frame = cv2.cvtColor(
            frame.cpu().numpy().astype(np.uint8).transpose(1, 2, 0),
            cv2.COLOR_RGB2BGR,
        ).copy()

        if boxes.shape[0] == 0:
            cv2.imwrite(filename, init_frame)
            return

        for j in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[j]
            cx, cy = keypoints[j, 0, :2]

            cv2.rectangle(
                init_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                2,
            )

            # Draw confidence score
            cv2.putText(
                init_frame,
                f"{scores[j].item():.2f}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

            # Draw center point (nose)
            cv2.circle(init_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

            # Draw skeleton if available
            if keypoints.shape[0] > 0:
                kpts = keypoints[j].cpu().numpy()  # Shape: (K, 3)

                # Draw skeleton connections
                for start_idx, end_idx, color in SKELETON_CONNECTIONS:
                    if start_idx < kpts.shape[0] and end_idx < kpts.shape[0]:
                        start_pt = kpts[start_idx, :2]
                        start_conf = kpts[start_idx, 2]
                        end_pt = kpts[end_idx, :2]
                        end_conf = kpts[end_idx, 2]
                        if (
                            start_conf > visibility_threshold
                            and end_conf > visibility_threshold
                        ):
                            cv2.line(
                                init_frame,
                                (int(start_pt[0]), int(start_pt[1])),
                                (int(end_pt[0]), int(end_pt[1])),
                                color,
                                2,
                            )

                # Draw keypoints
                for k in range(keypoints.shape[1]):
                    kpt = keypoints[j, k, :]
                    if kpt[2] > visibility_threshold:
                        cv2.circle(
                            init_frame, (int(kpt[0]), int(kpt[1])), 3, (0, 0, 255), -1
                        )

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv2.imwrite(filename, init_frame)
