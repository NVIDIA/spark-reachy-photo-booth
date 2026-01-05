# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from pydantic import BaseModel, Field, SkipValidation, model_validator

##############################
# Animation models
##############################


class JointData1D(BaseModel):
    joints: str
    frames: list[float]

    @model_validator(mode="after")
    def validate_joints_and_frames(self):
        if self.joints == "":
            raise ValueError("Joints are empty. Please, provide at least one joint.")
        if len(self.frames) <= 0:
            raise ValueError("Frames are empty. Please, provide at least one frame.")
        return self


class JointData2D(BaseModel):
    joints: list[str]
    frames: list[list[float]]

    @model_validator(mode="after")
    def validate_joints_and_frames(self):
        if not self.joints:
            raise ValueError("Joints are empty. Please, provide at least one joint.")
        if not self.frames:
            raise ValueError("Frames are empty. Please, provide at least one frame.")
        if len(self.frames[0]) != len(self.joints):
            raise ValueError(
                f"Number of values in one frame and number of joints must match. "
                f"Provided {len(self.frames[0])} values and {len(self.joints)} joints."
            )
        return self


class AnimationData(BaseModel):
    body_angle: JointData1D | None = None
    r_antenna_angle: JointData1D | None = None
    l_antenna_angle: JointData1D | None = None

    head_rotation: JointData2D | None = None
    head_position: JointData2D | None = None

    n_frames: SkipValidation[int] = 0

    @model_validator(mode="after")
    def validate_joints(self):
        if (
            not self.body_angle
            and not self.r_antenna_angle
            and not self.l_antenna_angle
            and not self.head_rotation
            and not self.head_position
        ):
            raise ValueError("Please, provide at least one joint data.")
        return self

    @model_validator(mode="after")
    def validate_n_frames(self):
        # Get all joints
        joint_attrs = [
            self.body_angle,
            self.r_antenna_angle,
            self.l_antenna_angle,
            self.head_rotation,
            self.head_position,
        ]

        # Get their frame counts
        n_frames = {len(j.frames) for j in joint_attrs if j is not None}

        if not n_frames:
            raise ValueError("Please, provide at least one frame.")

        if len(n_frames) > 1:
            raise ValueError("All joint data must have the same number of frames.")

        self.n_frames = n_frames.pop()
        return self


class Animation(BaseModel):
    frame_rate: int = Field(default=30, ge=1, lt=100)
    data: AnimationData

    @classmethod
    def from_json_file(cls, json_file: Path) -> "Animation":
        with open(json_file) as f:
            return cls.model_validate_json(f.read())
