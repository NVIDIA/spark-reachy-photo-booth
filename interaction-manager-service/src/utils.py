# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum

from workmesh.messages import Robot as RobotProto

type Degrees = float


@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0


class KeyPosition(Enum):
    """Key position identifier for the interaction manager service."""

    SCREEN = "screen"
    USER = "user"
    PRINTER = "printer"


class Robot(Enum):
    """Robot identifier for the interaction manager service."""

    RESEARCHER = "researcher"

    def to_proto(self) -> RobotProto:
        if self == Robot.RESEARCHER:
            return RobotProto.RESEARCHER
        raise ValueError(f"Invalid robot: {self}")
