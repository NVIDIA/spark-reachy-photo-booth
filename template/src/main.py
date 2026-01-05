# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, NoReturn

from configuration import MyServiceConfig
from workmesh.config import load_config
from workmesh.service import Service
from workmesh.service_executor import ServiceExecutor

from workmesh import Ping, ping_topic, produces, subscribe


class MyService(Service):
    def __init__(self, config: MyServiceConfig | None = None) -> None:
        super().__init__(config)
        self.create_task(self.heartbeat())
        self.create_task(self.send_ping())  # pyright: ignore[reportAttributeAccessIssue]

    @subscribe(ping_topic)
    async def on_ping(self, message: Ping) -> None:
        self.logger.info("Ping message received")

    @produces(ping_topic)
    async def send_ping(self) -> AsyncGenerator[Ping, Any]:
        yield Ping(ping_id="123", content="Hello world!")
        self.logger.info("Sent ping")

        await self.publish(
            ping_topic, message=Ping(ping_id="456", content="Hello world!")
        )
        self.logger.info("Sent another ping")

    async def heartbeat(self) -> NoReturn:
        while True:
            self.logger.info("I am alive")
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self.logger.info("I am dying!")
        await super().stop()


async def main() -> None:
    config = load_config(MyServiceConfig)
    await ServiceExecutor([MyService(config)]).run()


if __name__ == "__main__":
    asyncio.run(main())
