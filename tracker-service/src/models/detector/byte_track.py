# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import torch
from constants import NUM_KEYPOINTS
from models.detector.base import BaseDetector, DetectionResult
from models.detector.detectron2 import Detectron2Detector
from pydantic import BaseModel, Field
from yolox.tracker.byte_tracker import BYTETracker

np.float = float  # type: ignore (deprecated in NumPy 1.20)
warnings.filterwarnings("ignore")


class ByteTrackConfig(BaseModel):
    track_buffer: int = Field(
        default=2000,
        description="The number of frames to keep lost tracks.",
        ge=30,
        le=10000,
    )
    track_thresh: float = Field(
        default=0.5,
        description="The confidence threshold for tracking.",
        ge=0.0,
        le=1.0,
    )
    match_thresh: float = Field(
        default=0.8, description="The matching threshold for tracking.", ge=0.0, le=1.0
    )
    min_box_area: float = Field(
        default=100,
        description="The minimum area of a bounding box to be considered.",
        gt=0.0,
    )
    mot20: bool = Field(
        default=False, description="Whether to use MOT20 dataset for training."
    )


class ByteTrackDetector(BaseDetector):
    """ByteTrack-based human detector for video processing.

    This class provides human detection capabilities using ByteTrack models,
    optimized for processing video frames and returning detection results.
    """

    def __init__(
        self,
        model_name: str = "detectron2",
        device: str = "auto",
        confidence_threshold: float = 0.98,
        iou_threshold: float = 0.5,
        track_buffer: int = 2000,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        min_box_area: float = 100,
        mot20: bool = False,
    ):
        """Initialize the ByteTrack detector.

        Args:
            model_name (str): The underlying detector model to use.
            device (str): The device to use for inference.
            confidence_threshold (float): The confidence threshold for detection.
            iou_threshold (float): The IoU threshold for non-maximum suppression.
            track_buffer (int): The number of frames to keep lost tracks.
            track_thresh (float): The confidence threshold for tracking.
            match_thresh (float): The matching threshold for tracking.
            min_box_area (float): The minimum area of a bounding box to be considered.
            mot20 (bool): Whether to use MOT20 dataset for training.

        """
        super().__init__()
        if model_name == "detectron2":
            self.detector = Detectron2Detector(
                device=device,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
            )
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        self.model = BYTETracker(
            ByteTrackConfig(
                track_buffer=track_buffer,
                track_thresh=track_thresh,
                match_thresh=match_thresh,
                min_box_area=min_box_area,
                mot20=mot20,
            )
        )

    def _preprocess_video(self, video: torch.Tensor) -> np.ndarray:
        """Preprocess video tensor for inference.

        Args:
            video (torch.Tensor): Video tensor of shape (T, C, H, W)

        Returns:
            np.ndarray: Preprocessed video as numpy array in BGR format
                        (T, H, W, C)

        """
        video_processed = video.float()
        video_processed = video_processed[:, [2, 1, 0], :, :]  # from rgb to bgr

        if video_processed.max() <= 1.0:
            video_processed = video_processed * 255  # 0.0-1.0 to 0-255

        video_np: np.ndarray = (
            video_processed.byte().permute(0, 2, 3, 1).cpu().numpy()
        )  # (T, H, W, C)
        return video_np

    def _compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Compute IoU between two boxes (x1, y1, x2, y2 format)."""
        x1 = max(box1[0].item(), box2[0].item())
        y1 = max(box1[1].item(), box2[1].item())
        x2 = min(box1[2].item(), box2[2].item())
        y2 = min(box1[3].item(), box2[3].item())

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return float(intersection) / float(union) if union > 0 else 0.0

    def _match_tracks_to_detections(
        self,
        track_boxes: torch.Tensor,
        detection_boxes: torch.Tensor,
    ) -> list[int]:
        """
        Match tracked boxes to original detection boxes using IoU.

        Args:
            track_boxes: Tracked boxes (N_tracks, 4)
            detection_boxes: Original detection boxes (N_detections, 4)

        Returns:
            List of detection indices for each track (-1 if no match)
        """
        matches = []
        for track_box in track_boxes:
            best_iou = 0.0
            best_idx = -1
            for det_idx, det_box in enumerate(detection_boxes):
                iou = self._compute_iou(track_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = det_idx
            # Use a threshold to ensure good matches
            matches.append(best_idx if best_iou > 0.5 else -1)
        return matches

    def run(self, video: torch.Tensor) -> DetectionResult:
        video_np = self._preprocess_video(video)
        T, H, W, _ = video_np.shape
        assert T == 1, "Only single frame input is supported"

        detection_result = self.detector.run(video)
        if detection_result.boxes.shape[0] == 0:
            return DetectionResult(
                boxes=torch.zeros((0, 4)),
                scores=torch.zeros((0,)),
                instance_ids=torch.zeros((0,)),
                keypoints=torch.zeros((0, NUM_KEYPOINTS, 3)),
            )

        bbox_bytetrack = torch.concat(
            (
                detection_result.boxes[:, :4],
                detection_result.scores.unsqueeze(1),
            ),
            dim=1,
        )  # shape: (n, 5)

        tracks = self.model.update(bbox_bytetrack, img_info=(H, W), img_size=(H, W))

        if len(tracks) == 0:
            return DetectionResult(
                boxes=torch.zeros((0, 4)),
                scores=torch.zeros((0,)),
                instance_ids=torch.zeros((0,)),
                keypoints=torch.zeros((0, NUM_KEYPOINTS, 3)),
            )

        instance_ids = torch.tensor([i.track_id for i in tracks])
        bboxes = torch.stack(
            [
                torch.tensor(
                    [
                        *i.tlbr,  # x1, y1, x2, y2
                    ]
                )
                for i in tracks
            ]
        )
        scores = torch.tensor([i.score for i in tracks])

        # reproject outside points to inside points due to kalman filter
        bboxes[:, 0] = torch.clamp(bboxes[:, 0], 0, W)
        bboxes[:, 1] = torch.clamp(bboxes[:, 1], 0, H)
        bboxes[:, 2] = torch.clamp(bboxes[:, 2], 0, W)
        bboxes[:, 3] = torch.clamp(bboxes[:, 3], 0, H)

        # Match tracks to original detections to get correct keypoints
        track_to_detection = self._match_tracks_to_detections(
            bboxes, detection_result.boxes
        )

        # Reorder keypoints to match tracks
        matched_keypoints = []
        num_keypoints = detection_result.keypoints.shape[1]
        for det_idx in track_to_detection:
            if det_idx >= 0:
                matched_keypoints.append(detection_result.keypoints[det_idx])
            else:
                # No match found - use zero keypoints
                matched_keypoints.append(torch.zeros((num_keypoints, 3)))

        keypoints = torch.stack(matched_keypoints)  # shape: (N_tracks, K, 3)

        return DetectionResult(
            boxes=bboxes,
            scores=scores,
            instance_ids=instance_ids,
            keypoints=keypoints,
        )
