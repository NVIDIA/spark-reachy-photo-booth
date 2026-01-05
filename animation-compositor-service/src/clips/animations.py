# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, fields
from typing import Annotated, Literal

import numpy as np
from models import Range
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

type Degrees = float

LINEAR_JOINT_NAMES = [
    "body_angle",
    "r_antenna_angle",
    "l_antenna_angle",
    "head_position_x",
    "head_position_y",
    "head_position_z",
]
SPHERICAL_JOINT_NAMES = ["head_rotation"]
JOINT_NAMES = LINEAR_JOINT_NAMES + SPHERICAL_JOINT_NAMES

MAX_DELTA_PER_FRAME_DEFAULT = 100.0


@dataclass()
class EulerAngles:
    """Euler angles."""

    roll: Degrees = 0.0
    pitch: Degrees = 0.0
    yaw: Degrees = 0.0

    def to_list(self) -> list[float]:
        return [self.roll, self.pitch, self.yaw]

    def __str__(self) -> str:
        return f"{{'roll': {self.roll}, 'pitch': {self.pitch}, 'yaw': {self.yaw}}}"

    def __eq__(self, other: object) -> bool:
        """Compare EulerAngles with floating point tolerance."""
        if not isinstance(other, EulerAngles):
            return False
        return (
            math.isclose(self.roll, other.roll, rel_tol=1e-9, abs_tol=1e-9)
            and math.isclose(self.pitch, other.pitch, rel_tol=1e-9, abs_tol=1e-9)
            and math.isclose(self.yaw, other.yaw, rel_tol=1e-9, abs_tol=1e-9)
        )


@dataclass()
class Frame:
    """Single animation frame."""

    # Linear interpolation
    body_angle: Degrees | None = None
    r_antenna_angle: Degrees | None = None
    l_antenna_angle: Degrees | None = None
    head_position_x: float | None = None
    head_position_y: float | None = None
    head_position_z: float | None = None

    # Spherical linear interpolation
    head_rotation: EulerAngles | None = None

    def __post_init__(self):
        """Post init."""

        field_names = fields(Frame)
        self.weight_mask: np.ndarray = np.zeros(len(field_names))

        # Init weight mask
        for i, attr in enumerate(field_names):
            if getattr(self, attr.name) is not None:
                self.weight_mask[i] = 1.0

    def __str__(self) -> str:
        return (
            "{"
            f"'body_angle': {self.body_angle}, "
            f"'r_antenna_angle': {self.r_antenna_angle}, "
            f"'l_antenna_angle': {self.l_antenna_angle}, "
            f"'head_position_x': {self.head_position_x}, "
            f"'head_position_y': {self.head_position_y}, "
            f"'head_position_z': {self.head_position_z}, "
            f"'head_rotation': {self.head_rotation}"
            "}"
        )

    def __eq__(self, other: object) -> bool:
        """Compare Frame with floating point tolerance."""
        if not isinstance(other, Frame):
            return False

        # Compare all float fields with tolerance
        for joint in LINEAR_JOINT_NAMES:
            self_val = getattr(self, joint)
            other_val = getattr(other, joint)

            # Handle None values
            if self_val is None and other_val is None:
                continue
            if self_val is None or other_val is None:
                return False

            # Compare with tolerance
            if not math.isclose(self_val, other_val, rel_tol=1e-9, abs_tol=1e-9):
                return False

        # Compare head_rotation (EulerAngles)
        return self.head_rotation == other.head_rotation

    def get_joint_names(self) -> list[str]:
        return [field.name for field in fields(Frame)]

    def blend_to_frame(self, other_frame: "Frame", weight: float) -> "Frame":
        """
        Blend two frames.

        Parameters:
            other_frame (Frame): The target frame to blend towards.
            weight (float): The blend weight, between 0.0 (only self)
                    and 1.0 (only other_frame).

        Returns:
            Frame: The blended frame.
        """

        # Blend joints with linear interpolation
        blended_joints: dict[str, Degrees | EulerAngles | None] = {}
        for joint in LINEAR_JOINT_NAMES:
            blended_joints[joint] = interpolate(
                joint, self, other_frame, weight, "lerp"
            )

        # Blend joints with spherical linear interpolation
        for joint in SPHERICAL_JOINT_NAMES:
            blended_joints[joint] = interpolate(joint, self, other_frame, weight, "slerp")  # noqa: E501 #fmt: skip

        return Frame(**blended_joints)  # type: ignore

    def additive_blend(self, other_frame: "Frame") -> "Frame":
        """
        Additively blend this frame with another frame.

        For each joint, if both frames have a value, their values are added together.
        For linear joints, this is a simple sum. For spherical joints (rotations), the
        rotations are composed (multiplied). If only one frame has a value for a joint,
        that value is used.

        Parameters:
            other_frame (Frame): The frame to add to this frame.

        Returns:
            Frame: The result of the additive blend.
        """

        added_joints: dict[str, Degrees | EulerAngles | None] = {}

        # Add simple joints
        for joint in LINEAR_JOINT_NAMES:
            joint1 = getattr(self, joint)
            joint2 = getattr(other_frame, joint)

            if joint1 is not None and joint2 is not None:
                added_joints[joint] = joint1 + joint2
            else:
                added_joints[joint] = joint1 if joint1 is not None else joint2

        # Add spherical joints
        for joint in SPHERICAL_JOINT_NAMES:
            joint1 = getattr(self, joint)
            joint2 = getattr(other_frame, joint)

            if joint1 is not None and joint2 is not None:
                r1 = R.from_euler("zyx", joint1.to_list(), degrees=True)
                r2 = R.from_euler("zyx", joint2.to_list(), degrees=True)

                # Compose rotations
                r = r2 * r1

                # Convert to euler angles
                new_angles = r.as_euler("zyx", degrees=True)

                added_joints[joint] = EulerAngles(
                    roll=new_angles[0], pitch=new_angles[1], yaw=new_angles[2]
                )
            else:
                added_joints[joint] = joint1 if joint1 is not None else joint2

        return Frame(**added_joints)  # type: ignore

    def subtract_frame(self, other_frame: "Frame") -> "Frame":
        """
        Subtract a frame from another frame (self - other_frame).
        If a joint is not present in the other frame, it is left as is.
        If a joint is not present in the self frame, it is set to None.

        Parameters:
            other_frame (Frame): The frame to subtract from this frame.
        """
        final_joints: dict[str, Degrees | EulerAngles | None] = {}

        # Subtract simple joints
        for joint in LINEAR_JOINT_NAMES:
            joint1 = getattr(self, joint)
            joint2 = getattr(other_frame, joint)

            if joint1 is not None and joint2 is not None:
                final_joints[joint] = joint1 - joint2
            elif joint1 is not None:
                # Leave as self value if we don't subtract
                final_joints[joint] = joint1

        # Subtract spherical joints
        for joint in SPHERICAL_JOINT_NAMES:
            joint1 = getattr(self, joint)
            joint2 = getattr(other_frame, joint)

            if joint1 is not None and joint2 is not None:
                # Convert both to Rotation (quaternion) objects
                r1 = R.from_euler("zyx", joint1.to_list(), degrees=True)
                r2 = R.from_euler("zyx", joint2.to_list(), degrees=True)

                # Subtract orientations
                r_sub = r1 * r2.inv()

                # Convert back to Euler angles
                new_angles = r_sub.as_euler("zyx", degrees=True)

                final_joints[joint] = EulerAngles(
                    roll=new_angles[0], pitch=new_angles[1], yaw=new_angles[2]
                )
            elif joint1 is not None:
                # Leave as self value if we don't subtract
                final_joints[joint] = joint1
        return Frame(**final_joints)  # type: ignore

    @staticmethod
    def reference_pose() -> "Frame":
        """Return the reference pose of the frame."""
        frame_kwargs = {joint: 0.0 for joint in LINEAR_JOINT_NAMES} | {
            joint: EulerAngles() for joint in SPHERICAL_JOINT_NAMES
        }
        return Frame(**frame_kwargs)  # type: ignore

    def clamp_frame(self, joint_limits: dict[str, Range]) -> None:
        """Clamp all joint values in the frame to their specified joint limits."""

        for joint_name in self.get_joint_names():
            joint_value = getattr(self, joint_name)
            if isinstance(joint_value, float) and joint_name in joint_limits:
                joint_value = joint_limits[joint_name].clamp(joint_value)
                setattr(self, joint_name, joint_value)
            elif isinstance(joint_value, EulerAngles):
                for angle_name in ["roll", "pitch", "yaw"]:
                    angle_joint_name = f"{joint_name}_{angle_name}"
                    angle_value = getattr(joint_value, angle_name)
                    if angle_value is not None and angle_joint_name in joint_limits:  # noqa: E501 #fmt: skip
                        angle_value = joint_limits[angle_joint_name].clamp(angle_value)  # noqa: E501 #fmt: skip
                        setattr(joint_value, angle_name, angle_value)

    def clip_frame_by_max_delta(
        self, previous_frame: "Frame", max_delta_per_frame: dict[str, float]
    ) -> tuple["Frame", bool]:
        """
        Smooth frame transitions by limiting the maximum change per frame.
        This prevents sudden jumps by gradually moving toward the target position.
        """

        smoothed_joints: dict[str, float | EulerAngles | None] = {}
        was_clipped = False

        # Process linear joints
        for joint_name in LINEAR_JOINT_NAMES:
            target_value = getattr(self, joint_name)
            previous_value = getattr(previous_frame, joint_name)

            assert target_value is not None
            assert previous_value is not None

            # Calculate delta
            delta = target_value - previous_value
            max_delta = max_delta_per_frame.get(joint_name, MAX_DELTA_PER_FRAME_DEFAULT)

            # Check if delta is greater than max delta
            if abs(delta) > max_delta:
                was_clipped = True

            # Clip delta to maximum allowed change
            clipped_delta = np.clip(delta, -max_delta, max_delta)

            # Apply clipped delta to previous value
            smoothed_joints[joint_name] = previous_value + clipped_delta

        # Process spherical joints
        for joint_name in SPHERICAL_JOINT_NAMES:
            target_value = getattr(self, joint_name)
            previous_value = getattr(previous_frame, joint_name)

            assert isinstance(target_value, EulerAngles)
            assert isinstance(previous_value, EulerAngles)

            max_rotation_delta = max_delta_per_frame.get(joint_name, 5.0)

            smoothed_angles = {}
            for angle_name in ["roll", "pitch", "yaw"]:
                target_angle = getattr(target_value, angle_name)
                previous_angle = getattr(previous_value, angle_name)

                # Calculate delta
                delta = target_angle - previous_angle

                # Normalize delta to [-180, 180] range to handle wrapping
                while delta > 180:
                    delta -= 360
                while delta < -180:
                    delta += 360

                # Check if delta is greater than max delta
                if abs(delta) > max_rotation_delta:
                    was_clipped = True

                # Clip delta
                clipped_delta = np.clip(delta, -max_rotation_delta, max_rotation_delta)

                # Apply clipped delta
                smoothed_angles[angle_name] = previous_angle + clipped_delta

            smoothed_joints[joint_name] = EulerAngles(**smoothed_angles)

        return Frame(**smoothed_joints), was_clipped  # type: ignore

    def fill_missing_joints(self) -> "Frame":
        """Fill missing joints with reference pose."""

        filled_joints: dict[str, float | EulerAngles | None] = {}

        for joint in self.get_joint_names():
            if getattr(self, joint) is None:
                filled_joints[joint] = getattr(Frame.reference_pose(), joint)
            else:
                filled_joints[joint] = getattr(self, joint)

        return Frame(**filled_joints)  # type: ignore


@dataclass()
class Animation:
    """Animation frames."""

    frames: list[Frame]

    def eval_frame(self, animation_duration: float, time: float) -> Frame:
        """Evaluate a frame of an animation at a specific time."""
        assert animation_duration > 0.0
        assert len(self.frames) > 0

        frame = Frame()
        n_frames = len(self.frames)

        frame_rate = (n_frames - 1) / animation_duration
        lower_frame_index = int(np.clip(math.floor(time * frame_rate), 0, n_frames - 1))
        upper_frame_index = int(np.clip(math.ceil(time * frame_rate), 0, n_frames - 1))

        # Get the specific frame
        if lower_frame_index == upper_frame_index:
            frame = self.frames[lower_frame_index]
        # Interpolate frames
        else:
            delta_time = (upper_frame_index / frame_rate) - (
                lower_frame_index / frame_rate
            )
            relative_time = time - lower_frame_index / frame_rate

            weight = relative_time / delta_time

            # Blend the frames using weights
            lower_frame = self.frames[lower_frame_index]
            upper_frame = self.frames[upper_frame_index]

            frame = lower_frame.blend_to_frame(upper_frame, weight)

        return frame

    def eval_frame_loop(
        self,
        animation_duration: float,
        clip_time: float,
        loop: bool,
        loop_overlap: float,
    ) -> Frame:
        """Evaluate a frame of an animation with/without looping."""

        if loop and animation_duration > loop_overlap:
            time_in_loop = clip_time % (animation_duration - loop_overlap)

            if time_in_loop < loop_overlap:
                frame = self.eval_frame(animation_duration, clip_time)
                frame_other = self.eval_frame(animation_duration, loop_overlap)

                weight = time_in_loop / (loop_overlap * 2)
                if clip_time < loop_overlap:
                    weight = time_in_loop / (loop_overlap * 2) + 0.5

                return frame.blend_to_frame(frame_other, weight)

        return self.eval_frame(animation_duration, clip_time)


def blend_multiple_frames(
    frames: list[Frame],
    weights: list[dict[str, float]],
    total_weights: dict[str, float],
) -> Frame:
    """Blend multiple frames with weights."""

    blended_joints: dict[str, float | EulerAngles] = {}
    current_weight = 0.0

    # Blend frames
    for frame, weight in zip(frames, weights, strict=True):
        # Blend linear joints (antennas, body rotation, head translation)
        for joint in LINEAR_JOINT_NAMES:
            joint_value = getattr(frame, joint)
            if joint_value is not None and weight.get(joint, 0.0) > 0:
                blended_joints[joint] = (
                    blended_joints.get(joint, 0.0) + joint_value * weight[joint]
                )

        # Blend spherical joints (head rotation)
        for joint in SPHERICAL_JOINT_NAMES:
            joint_value = getattr(frame, joint)
            if joint_value is not None:
                joint_to_blend = blended_joints.get(joint)
                if joint_to_blend is None:
                    blended_joints[joint] = joint_value
                    current_weight = weight[joint]
                elif isinstance(joint_to_blend, EulerAngles) and isinstance(
                    joint_value, EulerAngles
                ):
                    interpolation_weight = 1 - current_weight / (
                        current_weight + weight[joint]
                    )
                    blended_joints[joint] = slerp(
                        joint_to_blend, joint_value, interpolation_weight
                    )
                    current_weight += weight[joint]

    # Blend head orientation with reference pose if total weight is not 1.
    # We can't blend in/out otherwise
    for joint in SPHERICAL_JOINT_NAMES:
        joint_to_blend = blended_joints.get(joint)
        if isinstance(joint_to_blend, EulerAngles) and total_weights[joint] < 1:
            blended_joints[joint] = slerp(
                EulerAngles(), joint_to_blend, total_weights[joint]
            )

    frame_kwargs = blended_joints
    return Frame(**frame_kwargs)  # type: ignore


def interpolate(
    joint: str,
    frame1: Frame,
    frame2: Frame,
    weight: float,
    interpolation_type: Annotated[str, Literal["lerp", "slerp"]] = "lerp",
) -> float | EulerAngles:
    interpolate_func = slerp if interpolation_type == "slerp" else lerp

    if getattr(frame1, joint) is not None and getattr(frame2, joint) is not None:
        interpolated_joint = interpolate_func(
            getattr(frame1, joint), getattr(frame2, joint), weight
        )
    else:
        interpolated_joint = (
            getattr(frame1, joint)
            if getattr(frame1, joint) is not None
            else getattr(frame2, joint)
        )
    return interpolated_joint


def lerp(frame1: float, frame2: float, weight: float) -> float:
    """Linear interpolation between two frames."""
    return frame1 * (1 - weight) + frame2 * weight


def slerp(frame1: EulerAngles, frame2: EulerAngles, weight: float) -> EulerAngles:
    """Slerp between two frames."""
    rots = R.from_euler(
        "zyx",
        [
            [frame1.roll, frame1.pitch, frame1.yaw],
            [frame2.roll, frame2.pitch, frame2.yaw],
        ],  # noqa: E501 #fmt: skip
        degrees=True,
    )
    slerp = Slerp([0, 1], rots)

    # Adjust weight for more linear angular progression in larger rotations
    angle = (rots[1] * rots[0].inv()).magnitude()
    if angle > 0.01:
        weight = max(0.0, min(1.0, weight ** (np.pi / (2 * angle))))

    new_angles = slerp(weight).as_euler("zyx", degrees=True)
    return EulerAngles(roll=new_angles[0], pitch=new_angles[1], yaw=new_angles[2])
