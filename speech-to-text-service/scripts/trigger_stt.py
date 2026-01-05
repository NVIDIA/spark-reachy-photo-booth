# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any, Literal

import typer
from workmesh.messages import Command, Service, ServiceCommand
from workmesh.service import Producer, Topic

from workmesh import service_command_topic


async def producer_send(topic: Topic, message: Any) -> None:
    async with Producer() as producer:
        await producer.publish(topic, message)


def main(status: Literal["on", "off"] = "on") -> None:
    message = ServiceCommand(
        command=Command.ENABLE if status == "on" else Command.DISABLE,
        target_service=Service.STT,
    )
    asyncio.run(producer_send(service_command_topic, message))


if __name__ == "__main__":
    typer.run(main)
