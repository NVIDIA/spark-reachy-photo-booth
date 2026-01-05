# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
from easing_curves import ease_in_out_back
from utils import Degrees, EulerAngle, RobotPosition
from workmesh.messages import Position2D


def look_at_animation(
    start_angle: Degrees,
    target: Position2D | Degrees,
    duration: float,
    frame_rate: int,
) -> list[Degrees]:
    """Look at a point in the world coordinate system. Rotate the base only"""

    if isinstance(target, Position2D):
        # Calculate the angle between the start and end positions
        target_angle = math.degrees(math.atan2(target.y, target.x))
    else:
        target_angle = target

    # Interpolation between the start and end positions
    times = np.linspace(0, duration, int(frame_rate * duration))

    angles: list[Degrees] = []
    for t in times:
        progress = t / duration
        angle = start_angle + (
            (target_angle - start_angle) * ease_in_out_back(progress)
        )
        angles.append(angle)
    return angles


def track_animation(
    current_robot_position: RobotPosition,
    position_x: float,
    position_y: float,
    slow_mode_distance_threshold: float,
    fast_mode_distance_threshold: float,
    slow_speed: float,
    fast_speed: float,
) -> RobotPosition | None:
    """Look at a point in the 2D plane of the camera."""
    # Center the user: transform coordinates from [0,1] to [-1,1]
    x = 2 * position_x - 1
    y = 2 * position_y - 1

    # Calculate the distance from the center of the camera
    distance = math.sqrt(x**2 + y**2)

    # If the user is too close to the center, don't move
    if distance < slow_mode_distance_threshold:
        return None

    # If the user is far from the center, move faster
    # TODO: adjust speed after fixing the camera fps
    speed = fast_speed if distance >= fast_mode_distance_threshold else slow_speed

    error = np.array([0, 0]) - np.array([x, y])
    head_rotation = current_robot_position.head_rotation.to_array() + np.array(
        [0.0, -speed * error[1], 0.0]
    )

    return RobotPosition(
        body_angle=current_robot_position.body_angle + speed * error[0],
        head_rotation=EulerAngle(
            roll=head_rotation[0],
            pitch=head_rotation[1],
            yaw=head_rotation[2],
        ),
    )
