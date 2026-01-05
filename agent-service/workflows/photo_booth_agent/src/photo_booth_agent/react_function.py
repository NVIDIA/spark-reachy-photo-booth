# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging

from langchain.prompts import ChatPromptTemplate
from nat.agent.react_agent.register import ReActAgentWorkflowConfig
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatRequest, ChatResponse, Usage
from nat.utils.type_converter import GlobalTypeConverter
from pydantic import Field

from photo_booth_agent.constant import SERVICE_NAME
from photo_booth_agent.state import StructuredReActGraphState

logger = logging.getLogger(SERVICE_NAME)


class PhotoBoothReactWorkflowConfig(ReActAgentWorkflowConfig, name="photo_booth_react"):
    """
    Configuration for the Photo Booth ReAct Agent Workflow.

    Extends the ReActAgent Workflow Config to add the ability
    to send intermediate steps to Kafka.
    """

    eval_mode: bool = Field(default=False, description="Whether to run in eval mode")
    allow_null_tool: bool = Field(
        default=True, description="Whether tool call can be null"
    )
    human_feedback_tool: str = Field(
        default="ask_human", description="The tool to use if the tool call is null"
    )
    summarize_every_n_turns: int = Field(
        default=5,
        description="The number of turns to summarize the conversation history",
    )
    summarize_timeout: int = Field(
        default=10,
        description="The timeout for the summarization LLM call",
    )
    greet_message: list[str] = Field(
        default=[
            "Hey there! Let's create an image.",
            "Hey you! Let me snap a photo of you!",
        ],
        description="Message variants to greet the user",
    )


@register_function(
    config_type=PhotoBoothReactWorkflowConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def photo_booth_react_function(
    config: PhotoBoothReactWorkflowConfig, builder: Builder
):
    from langchain.prompts import MessagesPlaceholder
    from langchain.schema import BaseMessage
    from langchain_core.messages import convert_to_messages
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph
    from nat.agent.base import AGENT_LOG_PREFIX
    from workmesh.messages import Robot

    from photo_booth_agent.callbacks import KafkaAsyncCallbackHandler
    from photo_booth_agent.output import StructuredAgentAction
    from photo_booth_agent.prompt import SYSTEM_PROMPT
    from photo_booth_agent.react_agent import PhotoBoothReactAgentGraph

    ########################################
    ####### SYSTEM PROMPT ##################
    ########################################
    prompt = SYSTEM_PROMPT if not config.system_prompt else config.system_prompt
    if config.additional_instructions:
        prompt += f" {config.additional_instructions}"

    if not PhotoBoothReactAgentGraph.validate_system_prompt(prompt):
        raise ValueError("Invalid system prompt")

    ########################################
    ####### PROMPT ########################
    ########################################
    prompt = ChatPromptTemplate(
        [
            ("system", prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
        ]
    )

    llm = await builder.get_llm(
        config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    tools: list[BaseTool] = await builder.get_tools(
        tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    if not tools:
        raise ValueError(f"No tools specified for ReAct Agent '{config.llm_name}'")

    try:
        llm = llm.bind_tools(tools)
        logger.info(f"{AGENT_LOG_PREFIX} Successfully bound tools to LLM")
    except Exception as ex:
        logger.exception(f"{AGENT_LOG_PREFIX} Failed to bind tools to LLM: {ex}")
        raise ex

    ########################################
    ####### AGENT #########################
    ########################################
    agent = PhotoBoothReactAgentGraph(
        llm=llm,
        prompt=prompt,
        tools=tools,
        use_tool_schema=config.include_tool_input_schema_in_tool_description,
        detailed_logs=config.verbose,
        retry_agent_response_parsing_errors=config.retry_agent_response_parsing_errors,
        parse_agent_response_max_retries=config.parse_agent_response_max_retries,
        tool_call_max_retries=config.tool_call_max_retries,
        pass_tool_call_errors_to_agent=config.pass_tool_call_errors_to_agent,
        eval_mode=config.eval_mode,
        structured_output=StructuredAgentAction.simple_schema(),
        max_history=config.max_history,
        allow_null_tool=config.allow_null_tool,
        human_feedback_tool=config.human_feedback_tool,
        summarize_every_n_turns=config.summarize_every_n_turns,
        summarize_timeout=config.summarize_timeout,
        greet_message=config.greet_message,
        callbacks=[
            KafkaAsyncCallbackHandler(robot_id=Robot.RESEARCHER),  # pyright: ignore[reportArgumentType]  # noqa: E501
        ],
    )

    graph: CompiledStateGraph = await agent.build_graph()

    ########################################
    ####### RESPONSE FUNCTION #############
    ########################################
    async def _response_fn(input_message: ChatRequest) -> ChatResponse:
        try:
            messages: list[BaseMessage] = convert_to_messages(
                [m.model_dump() for m in input_message.messages]
            )
            state = StructuredReActGraphState(messages=messages)
            state = await graph.ainvoke(
                state, config={"recursion_limit": (config.max_tool_calls + 1) * 2}
            )
            state = StructuredReActGraphState(**state)
            output_message = state.messages[-1]  # pylint: disable=E1136
            return ChatResponse.from_string(str(output_message.content), usage=Usage())

        except Exception as ex:
            logger.exception(
                f"{AGENT_LOG_PREFIX} ReAct Agent failed with exception: {ex}",
            )
            if config.verbose:
                return ChatResponse.from_string(str(ex), usage=Usage())
            return ChatResponse.from_string(
                "I seem to be having a problem.", usage=Usage()
            )

    ########################################
    ####### FUNCTION INFO #################
    ########################################
    async def _str_api_fn(input_message: str) -> str:
        try:
            input_message = json.loads(input_message)
        except json.JSONDecodeError:
            input_message = input_message

        if isinstance(input_message, list):
            oai_input = ChatRequest(messages=input_message)
        else:
            oai_input = GlobalTypeConverter.get().try_convert(
                input_message, to_type=ChatRequest
            )
        oai_output = await _response_fn(oai_input)

        return GlobalTypeConverter.get().try_convert(oai_output, to_type=str)

    yield FunctionInfo.from_fn(_str_api_fn, description=config.description)
