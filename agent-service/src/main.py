# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator, Awaitable, Callable

from configuration import AgentServiceConfig
from nat.builder.context import ContextState
from nat.data_models.interactive import (
    HumanResponse,
    HumanResponseText,
    InteractionPrompt,
)
from nat.runtime.loader import PluginTypes, discover_and_register_plugins, load_workflow
from workmesh.config import load_config
from workmesh.messages import (
    Command,
    ServiceCommand,
    UserUtterance,
    UserUtteranceStatus,
)
from workmesh.messages import (
    Service as ServiceName,
)
from workmesh.service import Service, subscribe
from workmesh.service_executor import ServiceExecutor

from workmesh import routed_user_utterance_topic, service_command_topic

WARM_UP_QUERY = "**Warm up call - respond with a simple `ignore-this-message`**"
GREET_USER_QUERY = "Hi"


def create_hitl_callback(
    agent_service: AgentService,
) -> Callable[[InteractionPrompt], Awaitable[HumanResponse]]:
    async def on_hitl(question: InteractionPrompt) -> HumanResponse:
        response = await agent_service.await_user_response()
        return HumanResponseText(text=response)

    return on_hitl


class AgentService(Service):
    def __init__(self, config: AgentServiceConfig) -> None:
        super().__init__(config)

        self.config = config
        self.logger.info("👷 Agent is ready!")

        discover_and_register_plugins(PluginTypes.ALL)
        self._all_messages: asyncio.Queue[UserUtterance] = asyncio.Queue()
        self._prompt_messages: asyncio.Queue[UserUtterance] = asyncio.Queue()
        self._agent_task: asyncio.Task[str | None] | None = None
        self._workflow_callable: Callable[[str], Awaitable[str | None]] | None = None
        self.create_task(self.process_text_from_user())
        self.create_task(self.setup_workflow())

    async def setup_workflow(self) -> None:
        """Initialize the workflow callable and warm it up."""

        self.cancel_agent: asyncio.Event = asyncio.Event()

        self.logger.info("Setting up workflow")
        async for callable in self.get_callable_for_workflow():
            self._workflow_callable = callable
            await callable(WARM_UP_QUERY)  # type: ignore[arg-type]
            with open("/dev/shm/ready", "w") as file:
                file.write("ready\n")

    async def stop(self) -> None:
        if self._agent_task:
            self.cancel_agent.set()
            with contextlib.suppress(asyncio.CancelledError):
                await self._agent_task
        await super().stop()

    @subscribe(service_command_topic)
    async def on_service_command(self, message: ServiceCommand) -> None:
        if message.target_service != ServiceName.AGENT:
            return
        if message.command != Command.RESTART:
            return
        self.logger.info("Received RESTART command, cancelling current workflow")
        self.cancel_agent.set()
        if self._agent_task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._agent_task
        self.cancel_agent.clear()

        if not self._workflow_callable:
            self.logger.warning("Workflow callable not ready for restart")
            return

        self._agent_task = asyncio.create_task(
            self._workflow_callable(GREET_USER_QUERY)  # type: ignore[arg-type]
        )

    @subscribe(routed_user_utterance_topic)
    async def on_user_utterance(self, message: UserUtterance) -> None:
        # Only process finished utterances
        if message.status != UserUtteranceStatus.USER_UTTERANCE_FINISHED:
            return

        if not (
            message.text.strip() != ""
            and abs(time.time() * 1000 - message.timestamp)
            < self.config.user_utterance_threshold
        ):
            self.logger.warning(f"User utterance not processed: {message.text}")
            return

        await self._all_messages.put(message)

    async def get_callable_for_workflow(
        self,
    ) -> AsyncGenerator[Callable[[str], Awaitable[str | None]], None]:
        """
        Creates an end-to-end async callable which can run a NAT workflow.

        Yields:
            Callable[[str], Awaitable[str | None]]: The callable that can be used to run
            queries through the workflow
        """
        async with load_workflow(self.config.nat_config.resolve()) as workflow:

            async def single_call(input_str: str) -> str | None:
                nat_context = ContextState.get()  # type: ignore[attr-defined]
                nat_context.user_input_callback.set(create_hitl_callback(self))
                if not input_str.strip() or input_str == WARM_UP_QUERY:
                    return None
                try:
                    async with workflow.run(input_str) as runner:
                        result_task: asyncio.Task[str] = asyncio.create_task(
                            runner.result(to_type=str)
                        )
                        cancel_task: asyncio.Task = asyncio.create_task(
                            self.cancel_agent.wait()
                        )
                        done, pending = await asyncio.wait(
                            [result_task, cancel_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in pending:
                            task.cancel()

                        if result_task in done:
                            result = result_task.result()
                            return result

                        elif cancel_task in done:
                            self.logger.info(
                                "Agent task cancelled, waiting for workflow to exit"
                            )
                            with contextlib.suppress(asyncio.CancelledError):
                                await result_task
                            return None
                except ValueError:
                    return None

            yield single_call

    async def process_text_from_user(self) -> None:
        while True:
            message = await self._all_messages.get()
            if self._agent_task is None or self._agent_task.done():
                continue

            await self._prompt_messages.put(message)

    async def await_user_response(self) -> str:
        threshold_ms = self.config.user_utterance_threshold
        while True:
            message = await self._prompt_messages.get()
            if (time.time() * 1000 - message.timestamp) <= threshold_ms:
                return message.text
            else:
                self.logger.warning(f"User response is too old: {message.text}")


async def main() -> None:
    config = load_config(AgentServiceConfig)
    service = AgentService(config)
    await ServiceExecutor([service]).run()


if __name__ == "__main__":
    asyncio.run(main())
