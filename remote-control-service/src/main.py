# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
from workmesh.service_executor import ServiceExecutor

from workmesh import (
    BaseConfig,
    RemoteControlCommand,
    Service,
    load_config,
    remote_control_command_topic,
)


class TriggerCommandRequest(BaseModel):
    command: str


class RemoteControlConfig(BaseConfig):
    commands: list[str] = Field(
        default_factory=list,
        description="List of available remote control commands",
        min_length=1,
        max_length=100,
    )

    @field_validator("commands")
    @classmethod
    def validate_commands(cls, v):
        if not isinstance(v, list):
            raise ValueError("Commands must be a list!")

        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Commands must not contain duplicates!")

        # Validate each command
        for command in v:
            if not isinstance(command, str):
                raise ValueError("All commands must be strings!")
            if not command.strip():
                raise ValueError("Commands cannot be empty or whitespace only!")
            if len(command) > 100:
                raise ValueError("Command names must be 100 characters or less!")

        return v


class RemoteControlService(Service):
    def __init__(self, config: RemoteControlConfig | None = None) -> None:
        super().__init__(config)
        self._config = config or RemoteControlConfig()
        self._app = FastAPI()

        @self._app.get("/", response_class=HTMLResponse)
        async def root() -> str:
            html_file = Path(__file__).parent / "index.html"
            return html_file.read_text()

        @self._app.get("/api/commands")
        async def api_commands() -> dict[str, list[str]]:
            return {"commands": self._config.commands}

        @self._app.post("/api/trigger-command")
        async def api_trigger_command(request: TriggerCommandRequest) -> dict[str, str]:
            self.logger.info(f"Remote control command: {request.command}")
            await self.publish(
                remote_control_command_topic,
                RemoteControlCommand(command=request.command),
            )
            return {"status": "success", "commandName": request.command}

        self.logger.info("🚀 Remote Control UI is available at: http://localhost:8888")
        self.logger.info("🎮 Ready to receive commands!")

    def get_app(self) -> FastAPI:
        return self._app


async def main() -> None:
    config = load_config(RemoteControlConfig)
    remote_control_service = RemoteControlService(config)

    uvicorn_config = uvicorn.Config(
        remote_control_service.get_app(), host="0.0.0.0", port=8888, log_level="warning"
    )
    server = uvicorn.Server(uvicorn_config)

    service_task = asyncio.create_task(ServiceExecutor([remote_control_service]).run())
    server_task = asyncio.create_task(server.serve())

    # Terminate the program when the service or the server crashes
    _, pending = await asyncio.wait(
        [service_task, server_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel any remaining tasks
    for task in pending:
        task.cancel()

    # Wait for cancelled tasks to finish
    await asyncio.gather(*pending, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
