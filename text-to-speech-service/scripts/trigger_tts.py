# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import uuid
from enum import Enum
from typing import Any

import typer
from workmesh.messages import HumanSpeechRequest
from workmesh.messages import Robot as RobotProto
from workmesh.service import Producer, Topic

from workmesh import human_speech_request_topic


class Robot(Enum):
    RESEARCHER = "researcher"

    def to_proto(self) -> RobotProto:
        if self == Robot.RESEARCHER:
            return RobotProto.RESEARCHER
        raise ValueError(f"Invalid robot: {self}")


async def producer_send(topic: Topic, message: Any) -> None:
    async with Producer() as producer:
        await producer.publish(topic, message)


def main(
    robot_id: Robot = Robot.RESEARCHER,
    script: str = "Hello, world!",
):
    asyncio.run(
        producer_send(
            human_speech_request_topic,
            HumanSpeechRequest(
                action_uuid=str(uuid.uuid4()),
                robot_id=robot_id.to_proto(),
                script=script,
            ),
        )
    )


if __name__ == "__main__":
    typer.run(main)
