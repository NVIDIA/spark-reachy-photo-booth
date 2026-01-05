# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from constants import NUM_KEYPOINTS, UNDEFINED_INSTANCE_ID
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from models.detector.base import BaseDetector, DetectionResult
from torchvision.ops import nms


class Detectron2Detector(BaseDetector):
    """Detectron2-based human detector for video processing.

    This class provides human detection capabilities using Detectron2 models,
    optimized for processing video frames and returning detection results.
    """

    def __init__(
        self,
        model_name: str = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml",
        device: str = "auto",
        confidence_threshold: float = 0.98,
        iou_threshold: float = 0.5,
    ):
        """Initialize the Detectron2 detector.

        Args:
            model_name (str): Detectron2 model config to use
            device (str): Device to run inference on ('auto', 'cpu', 'cuda')
            confidence_threshold (float): Minimum confidence score for detections
            iou_threshold (float): The IoU threshold for non-maximum suppression.
        """
        super().__init__()

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.human_class_id = 0  # COCO class ID for 'person'
        self._iou_threshold = iou_threshold
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Configure Detectron2
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_name))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.DEVICE = self.device

        # Create predictor
        self.predictor = DefaultPredictor(cfg)
        self._track_id = UNDEFINED_INSTANCE_ID

    def _preprocess_video(self, video: torch.Tensor) -> np.ndarray:
        """Preprocess video tensor for Detectron2 inference.

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

    def run(
        self,
        video: torch.Tensor,
    ) -> DetectionResult:
        video_np = self._preprocess_video(video)
        N, _, _, _ = video_np.shape  # noqa: N806
        assert N == 1, "Detectron2Detector only supports single frame input"

        frame_np = video_np[0, :, :, :]
        outputs = self.predictor(frame_np)

        if "instances" not in outputs or len(outputs["instances"]) == 0:
            return DetectionResult(
                boxes=torch.zeros((0, 4)),
                scores=torch.zeros((0,)),
                instance_ids=torch.zeros((0,)),
                keypoints=torch.zeros((0, NUM_KEYPOINTS, 3)),
            )

        instances = outputs["instances"]
        pred_boxes = instances.pred_boxes.tensor.cpu().numpy()  # shape: (N, 4)
        pred_scores = instances.scores.cpu().numpy()  # shape: (N,)
        pred_classes = instances.pred_classes.cpu().numpy()  # shape: (N,)

        # Filter for humans only and confidence threshold
        human_mask = (pred_classes == self.human_class_id) & (
            pred_scores >= self.confidence_threshold
        )
        human_boxes = pred_boxes[human_mask, :]  # shape: (M, 4)
        pred_keypoints = instances.pred_keypoints.cpu()  # shape: (N, NUM_KEYPOINTS, 3)
        pred_keypoints = pred_keypoints[human_mask]  # shape: (M, NUM_KEYPOINTS, 3)

        pred_scores = pred_scores[human_mask]  # shape: (M,)
        M = human_boxes.shape[0]  # noqa: N806
        frame_boxes = [
            torch.tensor(
                [
                    human_boxes[i, 0],
                    human_boxes[i, 1],
                    human_boxes[i, 2],
                    human_boxes[i, 3],
                ]
            )
            for i in range(M)
        ]
        frame_scores = [torch.tensor(pred_scores[i]) for i in range(M)]
        # Assign undefined instance IDs for each detection
        frame_instance_ids = [
            torch.tensor(self._track_id) for _ in range(M)
        ]  # shape: (M,)
        detection_boxes = torch.stack(frame_boxes).float()  # shape: (M, 4)
        detection_scores = torch.stack(frame_scores).float()  # shape: (M,)
        detection_instance_ids = torch.stack(frame_instance_ids).float()  # shape: (M,)
        detection_keypoints = pred_keypoints.float()  # shape: (M, NUM_KEYPOINTS, 3)
        boxes_indices = nms(detection_boxes, detection_scores, self._iou_threshold)
        return DetectionResult(
            boxes=detection_boxes[boxes_indices],
            scores=detection_scores[boxes_indices],
            instance_ids=detection_instance_ids[boxes_indices],
            keypoints=detection_keypoints[boxes_indices],
        )
