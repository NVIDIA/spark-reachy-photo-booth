# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

UNDEFINED_INSTANCE_ID = -1
NUM_KEYPOINTS = 17  # We use coco keypoints
NAME_TO_INDEX = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}
SKELETON_CONNECTIONS = [
    (
        NAME_TO_INDEX["nose"],
        NAME_TO_INDEX["left_eye"],
        (255, 0, 0),
    ),  # nose -> left_eye
    (
        NAME_TO_INDEX["nose"],
        NAME_TO_INDEX["right_eye"],
        (255, 0, 0),
    ),  # nose -> right_eye
    (
        NAME_TO_INDEX["left_eye"],
        NAME_TO_INDEX["left_ear"],
        (255, 0, 0),
    ),  # left_eye -> left_ear
    (
        NAME_TO_INDEX["right_eye"],
        NAME_TO_INDEX["right_ear"],
        (255, 0, 0),
    ),  # right_eye -> right_ear
    (
        NAME_TO_INDEX["nose"],
        NAME_TO_INDEX["left_shoulder"],
        (0, 255, 0),
    ),  # nose -> left_shoulder
    (
        NAME_TO_INDEX["nose"],
        NAME_TO_INDEX["right_shoulder"],
        (0, 255, 0),
    ),  # nose -> right_shoulder
    (
        NAME_TO_INDEX["left_shoulder"],
        NAME_TO_INDEX["left_elbow"],
        (0, 255, 0),
    ),  # left_shoulder -> left_elbow
    (
        NAME_TO_INDEX["left_elbow"],
        NAME_TO_INDEX["left_wrist"],
        (0, 255, 0),
    ),  # left_elbow -> left_wrist
    (
        NAME_TO_INDEX["right_shoulder"],
        NAME_TO_INDEX["right_elbow"],
        (0, 255, 255),
    ),  # right_shoulder -> right_elbow
    (
        NAME_TO_INDEX["right_elbow"],
        NAME_TO_INDEX["right_wrist"],
        (0, 255, 255),
    ),  # right_elbow -> right_wrist
    (
        NAME_TO_INDEX["left_shoulder"],
        NAME_TO_INDEX["left_hip"],
        (255, 255, 0),
    ),  # left_shoulder -> left_hip
    (
        NAME_TO_INDEX["right_shoulder"],
        NAME_TO_INDEX["right_hip"],
        (255, 255, 0),
    ),  # right_shoulder -> right_hip
    (
        NAME_TO_INDEX["left_hip"],
        NAME_TO_INDEX["left_knee"],
        (255, 0, 255),
    ),  # left_hip -> left_knee
    (
        NAME_TO_INDEX["left_knee"],
        NAME_TO_INDEX["left_ankle"],
        (255, 0, 255),
    ),  # left_knee -> left_ankle
    (
        NAME_TO_INDEX["right_hip"],
        NAME_TO_INDEX["right_knee"],
        (255, 128, 0),
    ),  # right_hip -> right_knee
    (
        NAME_TO_INDEX["right_knee"],
        NAME_TO_INDEX["right_ankle"],
        (255, 128, 0),
    ),  # right_knee -> right_ankle
    (
        NAME_TO_INDEX["left_hip"],
        NAME_TO_INDEX["right_hip"],
        (128, 128, 128),
    ),  # left_hip -> right_hip
]
