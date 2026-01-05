# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.common import TypedBaseModel
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.interactive import HumanPromptText, InteractionResponse
from pydantic import Field

from photo_booth_agent.constant import SERVICE_NAME
from photo_booth_agent.exception import NoResponseError
from photo_booth_agent.utils import wait_for_ack, wait_for_user_utterance_started

logger = logging.getLogger(SERVICE_NAME)


class HITLFunctionConfig(FunctionBaseConfig, name="hitl_function"):
    start_timeout: int = Field(
        default=10,
        description="The timeout in seconds for the user to start speaking.",
    )

    response_timeout: int = Field(
        default=60,
        description="The timeout in seconds for the user to respond to the prompt. \
    This timeout should be set to a big value to give enough time to respond. \
    We use it to avoid the tool being blocked if something goes wrong on another side.",
    )


class HITLFunctionInput(TypedBaseModel, name="hitl_function_input"):
    action_uuid: str = Field(
        default="",
        description="""
The action UUID of the tool call.
This field will be set by the framework.
You don't need to set it, just leave it empty.
""",
    )
    prompt: str = Field(
        default="",
        description="The prompt to ask the human.",
    )


@register_function(config_type=HITLFunctionConfig)
async def hitl_function(config: HITLFunctionConfig, builder: Builder):
    async def _arun(input: HITLFunctionInput) -> str:
        # Wait for the tool to be processed before starting the timeout
        await wait_for_ack(input.action_uuid)

        nat_context = Context.get()  # type: ignore
        user_input_manager = nat_context.user_interaction_manager

        if user_input_manager is None:
            raise RuntimeError("user input manager is not set")

        human_prompt_text = HumanPromptText(
            text=input.prompt, required=True, placeholder="<your response here>"
        )
        try:
            # Wait for the user to start speaking
            await wait_for_user_utterance_started(
                input.action_uuid, timeout=config.start_timeout
            )

            # Get the user's response
            try:
                response: InteractionResponse = await asyncio.wait_for(
                    user_input_manager.prompt_user_input(human_prompt_text),
                    timeout=config.response_timeout,
                )
            except TimeoutError as e:
                raise NoResponseError(
                    f"Tool timed out after {config.response_timeout} seconds."
                ) from e
        except TimeoutError as e:
            raise NoResponseError(
                f"Tool timed out after {config.start_timeout} seconds."
            ) from e
        return response.content.text.lower()  # type: ignore

    yield FunctionInfo.create(
        single_fn=_arun,
        input_schema=HITLFunctionInput,
        description=(
            "This function will be used to get the user's response to the prompt"
        ),
    )


class EndInteractionToolConfig(FunctionBaseConfig, name="end_interaction"):
    pass


class EndInteractionInput(TypedBaseModel, name="end_interaction_input"):
    action_uuid: str = Field(
        default="",
        description="""
The action UUID of the tool call.
This field will be set by the framework.
You don't need to set it, just leave it empty.
""",
    )
    message: str = Field(
        description="The message to end the interaction with the human.",
    )


@register_function(config_type=EndInteractionToolConfig)
async def end_interaction(config: EndInteractionToolConfig, builder: Builder):
    async def _arun(input: EndInteractionInput) -> str:
        await wait_for_ack(input.action_uuid)
        return "End of interaction."

    yield FunctionInfo.create(
        single_fn=_arun,
        input_schema=EndInteractionInput,
        description="End the interaction with the human.",
    )


class StartInteractionToolConfig(FunctionBaseConfig, name="start_interaction"):
    pass


class StartInteractionInput(TypedBaseModel, name="start_interaction_input"):
    action_uuid: str = Field(
        default="",
        description="""
The action UUID of the tool call.
This field will be set by the framework.
You don't need to set it, just leave it empty.
""",
    )


@register_function(config_type=StartInteractionToolConfig)
async def start_interaction(config: StartInteractionToolConfig, builder: Builder):
    async def _arun(input: StartInteractionInput) -> str:
        nat_context = Context.get()  # type: ignore
        user_input_manager = nat_context.user_interaction_manager

        if user_input_manager is None:
            raise RuntimeError("user input manager is not set")

        human_prompt_text = HumanPromptText(
            text="", required=True, placeholder="<your response here>"
        )
        response: InteractionResponse = await user_input_manager.prompt_user_input(
            human_prompt_text
        )
        return response.content.text.lower()  # type: ignore

    yield FunctionInfo.create(
        single_fn=_arun,
        input_schema=StartInteractionInput,
        description="Start the interaction with the human.",
    )
