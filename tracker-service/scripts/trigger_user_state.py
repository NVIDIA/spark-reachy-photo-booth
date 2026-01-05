# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any, Literal

import typer
from workmesh.messages import PresenceStatus, Robot, UserState
from workmesh.service import Producer, Topic

from workmesh import user_state_topic


async def producer_send(topic: Topic, message: Any) -> None:
    async with Producer() as producer:
        await producer.publish(topic, message)


def main(
    status: Literal["appeared", "disappeared"] = "appeared",
    robot_id: str = "researcher",
) -> None:
    message = UserState(
        robot_id=getattr(Robot, robot_id.upper()),
        status=getattr(PresenceStatus, f"USER_{status.upper()}"),
    )
    asyncio.run(producer_send(user_state_topic, message))


if __name__ == "__main__":
    typer.run(main)
