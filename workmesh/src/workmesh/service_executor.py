# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import signal
from asyncio import Future

from workmesh import Service


class ServiceExecutor:
    def __init__(self, services: list[Service]):
        self.services = services
        self._task: Future | None = None
        self._cancel_tasks: list[Future | None] = []

    def cleanup(self):
        for service in self.services:
            self._cancel_tasks.append(asyncio.create_task(service.stop()))

    async def cleanup_async(self):
        for service in self.services:
            await service.stop()

    async def run(self, handle_sigint: bool = True):
        loop = asyncio.get_event_loop()
        self._task = asyncio.gather(*[s.run() for s in self.services])
        if handle_sigint:
            loop.add_signal_handler(signal.SIGINT, lambda: self.cleanup())
            loop.add_signal_handler(signal.SIGTERM, lambda: self.cleanup())
        await self._task
