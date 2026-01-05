# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import logging
import random
import uuid
from json import JSONDecodeError
from typing import Literal

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from nat.agent.base import AGENT_CALL_LOG_MESSAGE, AGENT_LOG_PREFIX
from nat.agent.react_agent.agent import AgentDecision, HumanMessage, ReActAgentGraph
from pydantic import ValidationError
from workmesh.messages import Robot

from photo_booth_agent.constant import SERVICE_NAME
from photo_booth_agent.output import StructuredAgentAction
from photo_booth_agent.state import StructuredReActGraphState
from photo_booth_agent.utils import clean_reasoning

logger = logging.getLogger(SERVICE_NAME)

SYSTEM_LOG_PREFIX = "[SYSTEM]"


async def _trim_messages(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    max_history: int,
    timeout: int,
) -> list[BaseMessage]:
    logger.info(
        f"{SYSTEM_LOG_PREFIX} Trimming messages. "
        f"Current count: {len(messages)}, Max: {max_history}"
    )

    # ═══════════════════════════════════════════════════════════════
    # Step 1: Identify Messages to Keep vs Summarize
    # ═══════════════════════════════════════════════════════════════
    # Always keep:
    # - First message (usually system prompt or initial user input)
    # - Last N messages (recent conversation context)

    human_message = messages[0]

    # Calculate how many recent messages to keep
    # Keep enough to stay under max_history after adding summary
    keep_last_n = max(max_history - 2, 0)  # -2 for human + summary
    logger.info(f"{SYSTEM_LOG_PREFIX} Keeping last {keep_last_n} messages")

    messages_to_summarize = (
        messages[1:-keep_last_n] if len(messages) - keep_last_n - 2 > 0 else []
    )
    recent_messages = messages[-keep_last_n:]
    if len(recent_messages) == 0 or len(messages_to_summarize) == 0:
        return messages

    logger.info(
        f"{SYSTEM_LOG_PREFIX} Summarizing {len(messages_to_summarize)} messages. "
        f"Keeping human message + last {keep_last_n} messages"
    )

    # ═══════════════════════════════════════════════════════════════
    # Step 3: Generate Summary via LLM
    # ═══════════════════════════════════════════════════════════════
    # Format messages for the summarization LLM call

    try:
        summary = await llm.ainvoke(
            [
                HumanMessage(
                    content=f"""
/nothink
Summarize the following conversation history. Extract ONLY extremely important information that is critical for continuing the conversation.

{messages_to_summarize}

Include ONLY:
- User's name or identity if mentioned
- Specific user preferences, requirements, or constraints
- Critical context or decisions made
- Unfinished tasks or pending actions

Skip everything else (greetings, acknowledgments, small talk, completed tasks, etc.)

Format as a concise bullet list. Be extremely selective - if in doubt, leave it out.
"""  # noqa: E501
                )
            ],
            timeout=timeout,
        )
        summary_content = clean_reasoning(str(summary.content))
        logger.info(
            f"{SYSTEM_LOG_PREFIX} Generated summary: {summary_content[:100]}..."
        )

    except Exception as e:
        logger.error(f"{SYSTEM_LOG_PREFIX} Failed to generate summary: {e}")
        # Fallback: Return trimmed messages without summary to avoid infinite loop
        # Keep first message + recent messages to stay under max_history
        trimmed_messages = [human_message] + recent_messages
        logger.info(
            f"{SYSTEM_LOG_PREFIX} Returning trimmed messages without summary. "
            f"Messages: {len(messages)} → {len(trimmed_messages)}"
        )
        return trimmed_messages

    # ═══════════════════════════════════════════════════════════════
    # Step 4: Construct Trimmed Message List
    # ═══════════════════════════════════════════════════════════════
    # Structure: [first_message, summary_message, ...recent_messages]
    trimmed_messages = [
        human_message,
        AIMessage(content=f"[CONVERSATION SUMMARY]\n{summary_content}"),
    ] + recent_messages

    logger.info(
        f"{SYSTEM_LOG_PREFIX} Trimming complete. "
        f"Messages: {len(messages)} → {len(trimmed_messages)}"
    )

    return trimmed_messages


class PhotoBoothReactAgentGraph(ReActAgentGraph):
    def __init__(
        self,
        llm: BaseChatModel,
        prompt: ChatPromptTemplate,
        tools: list[BaseTool],
        use_tool_schema: bool = True,
        callbacks: list[AsyncCallbackHandler] | None = None,
        detailed_logs: bool = False,
        retry_agent_response_parsing_errors: bool = True,
        parse_agent_response_max_retries: int = 1,
        tool_call_max_retries: int = 1,
        pass_tool_call_errors_to_agent: bool = True,
        robot_id: Robot = Robot.RESEARCHER,
        eval_mode: bool = False,
        structured_output: dict | None = None,
        max_history: int = 15,
        allow_null_tool: bool = True,
        human_feedback_tool: str = "ask_human",
        summarize_every_n_turns: int = 5,
        summarize_timeout: int = 60,
        greet_tool: str = "greet_user",
        farewell_tool: str = "farewell_user",
        greet_message: list[str] | None = None,
        farewell_message: str = "Goodbye!",
    ):
        prompt = prompt.partial(structured_output=json.dumps(structured_output or {}))

        super().__init__(
            llm=llm,
            prompt=prompt,
            tools=tools,
            use_tool_schema=use_tool_schema,
            callbacks=callbacks,
            detailed_logs=detailed_logs,
            retry_agent_response_parsing_errors=retry_agent_response_parsing_errors,
            parse_agent_response_max_retries=parse_agent_response_max_retries,
            tool_call_max_retries=tool_call_max_retries,
            pass_tool_call_errors_to_agent=pass_tool_call_errors_to_agent,
        )
        self.robot_id = robot_id
        self.eval_mode = eval_mode
        self.max_history = max_history
        self.allow_null_tool = allow_null_tool
        self.human_feedback_tool = human_feedback_tool
        self.summarize_every_n_turns = summarize_every_n_turns
        self.llm = llm  # Store reference for message trimming
        self._iterations_since_summarization = 0  # Track when to summarize
        self.summarize_timeout = summarize_timeout
        self.greet_message = greet_message
        self.farewell_message = farewell_message
        self.greet_tool = greet_tool
        self.farewell_tool = farewell_tool
        if not self._get_tool(self.human_feedback_tool):
            raise ValueError(
                f"human_feedback_tool '{self.human_feedback_tool}' not found "
                f"in available tools: {list(self.tools_dict.keys())}"
            )

    async def agent_node(self, state: StructuredReActGraphState):  # pyright: ignore[reportIncompatibleMethodOverride]
        logger.info(f"{SYSTEM_LOG_PREFIX} Starting the ReAct Agent Node")

        # ═══════════════════════════════════════════════════════════════
        # RETRY LOOP - Attempt to parse LLM output
        # ═══════════════════════════════════════════════════════════════
        working_state = []
        try:
            for attempt in range(1, self.parse_agent_response_max_retries + 1):
                # ───────────────────────────────────────────────────────────
                # Step 1: Validate Input
                # ───────────────────────────────────────────────────────────
                if (
                    len(state.messages) == 0
                    or str(state.messages[-1].content).strip() == ""
                ):
                    raise RuntimeError(
                        f'{SYSTEM_LOG_PREFIX} No input received in state: "messages"'
                    )

                # ───────────────────────────────────────────────────────────
                # Step 3: Trim Message History (Every N Iterations)
                # ───────────────────────────────────────────────────────────

                self._iterations_since_summarization += 1

                if (
                    self._iterations_since_summarization % self.summarize_every_n_turns
                    == 0
                ):
                    state.messages = await _trim_messages(
                        llm=self.llm,
                        messages=state.messages,
                        max_history=self.max_history,
                        timeout=self.summarize_timeout,
                    )

                # ───────────────────────────────────────────────────────────
                # Step 4: Build Agent Scratchpad (for LLM context)
                # ───────────────────────────────────────────────────────────
                agent_scratchpad = working_state.copy()

                # ───────────────────────────────────────────────────────────
                # Step 5: Query the LLM
                # ───────────────────────────────────────────────────────────
                logger.info(f"{AGENT_LOG_PREFIX} Querying agent, attempt: {attempt}")
                output_message = await self.agent.ainvoke(
                    {
                        "chat_history": state.messages,  # All persistent history
                        "agent_scratchpad": agent_scratchpad,  # Retry errors only
                    },
                    RunnableConfig(callbacks=self.callbacks),  # type: ignore
                )

                if tool_calls := output_message.additional_kwargs.get("tool_calls", []):
                    tool_call = tool_calls[0]
                    function = tool_call.get("function", {})
                    function_name = function.get("name", None)
                    function_arguments = function.get("arguments", None)
                    agent_output = StructuredAgentAction(
                        thought=str(output_message.content),
                        tool=function_name,
                        tool_input=function_arguments,
                    )
                    original_content = agent_output.model_dump_json()
                    output_message.content = original_content
                else:
                    original_content = output_message.content
                    output_message.content = clean_reasoning(
                        str(original_content) or "*Empty response*"
                    )

                if self.detailed_logs:
                    logger.info(
                        AGENT_CALL_LOG_MESSAGE,
                        state.messages[-1].content,
                        output_message.content,
                    )

                # ───────────────────────────────────────────────────────────
                # Step 5: Parse and Validate LLM Output
                # ───────────────────────────────────────────────────────────
                try:
                    agent_output = StructuredAgentAction.model_validate_json(
                        str(output_message.content)
                    )

                    if not self.allow_null_tool and agent_output.tool is None:
                        logger.warning(
                            f"{AGENT_LOG_PREFIX} The tool cannot be null, "
                            " using the human feedback tool"
                        )

                        agent_output.tool = self.human_feedback_tool
                        agent_output.tool_input = {
                            "prompt": agent_output.final_answer or agent_output.thought,
                        }

                    # ───────────────────────────────────────────────────────
                    # Step 6: Add Agent Thought to Messages (Persistent)
                    # ───────────────────────────────────────────────────────
                    state.messages.append(
                        AIMessage(
                            content=output_message.content,
                        )
                    )

                    # ───────────────────────────────────────────────────────
                    # Step 7: Determine Next Step
                    # ───────────────────────────────────────────────────────
                    if agent_output.tool is None and agent_output.thought:
                        # ╔═══════════════════════════════════════════════╗
                        # ║ AGENT IS DONE - Final answer provided        ║
                        # ╚═══════════════════════════════════════════════╝
                        logger.debug(f"{AGENT_LOG_PREFIX} The agent is done.")
                        state.end = True
                    else:
                        # ╔═══════════════════════════════════════════════╗
                        # ║ TOOL CALL REQUIRED - Store action            ║
                        # ╚═══════════════════════════════════════════════╝
                        logger.debug(
                            f"{AGENT_LOG_PREFIX} The agent wants to call: {agent_output.tool}",  # noqa: E501
                        )
                        state.end = False
                        state.agent_scratchpad = [agent_output]

                    return state

                except ValidationError as ex:
                    # ───────────────────────────────────────────────────────
                    # Parsing Error - Retry with Error Feedback
                    # ───────────────────────────────────────────────────────
                    logger.debug(
                        f"{AGENT_LOG_PREFIX} Error parsing agent output\n"
                        f"Agent Output:\n{output_message.content}",
                    )

                    if attempt == self.parse_agent_response_max_retries:
                        raise RuntimeError("Failed to parse agent output") from ex

                    working_state.append(
                        AIMessage(
                            content=original_content or "**Empty response**",
                        )
                    )

                    error_message = f"Failed to parse agent output: {ex}.\n"
                    finish_reason = output_message.additional_kwargs.get(
                        "finish_reason"
                    )
                    if finish_reason == "length":
                        error_message += "The output is too long. Please try again with a shorter output."  # noqa: E501
                    else:
                        error_message += f"The correct format is: {json.dumps(StructuredAgentAction.simple_schema())}."  # noqa: E501

                    working_state.append(
                        HumanMessage(
                            content=error_message,
                        )
                    )
        except (ValidationError, RuntimeError) as e:
            logger.error(f"{AGENT_LOG_PREFIX} {e}")
            logger.info(f"{AGENT_LOG_PREFIX} Resetting the agent.")
            next_step = StructuredAgentAction(
                thought="I have run into an error. Communicating this to the user.",
                tool="ask_human",
                tool_input={
                    "prompt": "Seems like I have run into an error. I am resetting myself. Let's start over.",  # noqa: E501
                },
            )
            state.messages = [
                HumanMessage(content="The agent has run into an error."),
                AIMessage(content=next_step.model_dump_json()),
            ]
            state.end = False
            state.agent_scratchpad = [next_step]
            return state

    async def tool_node(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, state: StructuredReActGraphState
    ) -> StructuredReActGraphState:
        logger.debug(f"{AGENT_LOG_PREFIX} Starting the Tool Call Node")

        # ═══════════════════════════════════════════════════════════════
        # Step 1: Read Action from Ephemeral Scratchpad
        # ═══════════════════════════════════════════════════════════════
        if len(state.agent_scratchpad) == 0:
            raise RuntimeError('No tool input received in state: "agent_scratchpad"')

        agent_thoughts = state.agent_scratchpad[-1]

        # ───────────────────────────────────────────────────────────────
        # Step 2: Validate Tool Exists
        # ───────────────────────────────────────────────────────────────
        requested_tool = self._get_tool(agent_thoughts.tool or "")
        if not requested_tool:
            logger.warning(
                f"{AGENT_LOG_PREFIX} ReAct Agent wants to call tool "
                f"{agent_thoughts.tool} but no tool configured with that name. "
                f"Available: {list(self.tools_dict.keys())}",
            )
            # Add error to messages and clear scratchpad
            available_tools = list(self.tools_dict.keys())
            error_response = HumanMessage(
                content=f"Tool {agent_thoughts.tool} not found. Available: {available_tools}",  # noqa: E501
            )
            state.messages.append(error_response)
            return state

        # ───────────────────────────────────────────────────────────────
        # Step 3: Parse Tool Input
        # ───────────────────────────────────────────────────────────────
        action_uuid = str(uuid.uuid4())
        try:
            tool_input_str = str(agent_thoughts.tool_input).strip()
            logger.debug(f"{AGENT_LOG_PREFIX} Tool input string: {tool_input_str}")

            if tool_input_str == "None":
                tool_input_dict = {}
                logger.debug(f"{AGENT_LOG_PREFIX} Tool input is None — empty dict")
            else:
                try:
                    tool_input_dict = json.loads(tool_input_str)
                except JSONDecodeError:
                    tool_input_dict = ast.literal_eval(tool_input_str)
                logger.debug(f"{AGENT_LOG_PREFIX} Parsed structured tool input")

            tool_input_dict["action_uuid"] = action_uuid

        except (JSONDecodeError, ValueError, SyntaxError) as ex:
            logger.debug(
                f"{AGENT_LOG_PREFIX} Unable to parse structured tool input. "
                f"Using as string. Error: {ex}",
            )
            tool_input_dict = str(agent_thoughts.tool_input)

        # ───────────────────────────────────────────────────────────────
        # Step 4: Execute the Tool
        # ───────────────────────────────────────────────────────────────
        tool_response = await self._call_tool(
            requested_tool,
            tool_input_dict,
            RunnableConfig(callbacks=self.callbacks),  # type: ignore
            max_retries=self.tool_call_max_retries,
        )

        # ───────────────────────────────────────────────────────────────
        # Logging (if enabled)
        # ───────────────────────────────────────────────────────────────
        if self.detailed_logs:
            self._log_tool_response(
                requested_tool.name, tool_input_dict, str(tool_response.content)
            )

        # ───────────────────────────────────────────────────────────────
        # Error Handling
        # ───────────────────────────────────────────────────────────────
        if not self.pass_tool_call_errors_to_agent and tool_response.status == "error":
            logger.error(
                f"{AGENT_LOG_PREFIX} Tool {requested_tool.name} failed: "
                f"{tool_response.content}",
            )
            raise RuntimeError(f"Tool call failed: {tool_response.content}")

        # ═══════════════════════════════════════════════════════════════
        # Step 5: Add Response to Messages + Clear Scratchpad
        # ═══════════════════════════════════════════════════════════════
        state.messages.append(HumanMessage(content=str(tool_response.content)))
        state.agent_scratchpad = []
        return state

    async def start_interaction_node(self, state: StructuredReActGraphState):
        logger.info(f"{AGENT_LOG_PREFIX} Starting interaction")

        # Call greet_user tool
        greet_tool = self._get_tool(self.greet_tool)
        if not greet_tool:
            raise RuntimeError(f"{self.greet_tool} tool not found")

        await self._call_tool(
            greet_tool,
            {"action_uuid": str(uuid.uuid4())},
            RunnableConfig(callbacks=self.callbacks),  # type: ignore
            max_retries=self.tool_call_max_retries,
        )

        greeting_message = (
            "Hey there! Let's create an image."
            if self.greet_message is None
            else random.choice(self.greet_message)
        )

        state.messages.append(AIMessage(content=greeting_message))

        # Wait for human response
        human_tool = self._get_tool(self.human_feedback_tool)
        if not human_tool:
            raise RuntimeError(f"{self.human_feedback_tool} tool not found")

        tool_response = await self._call_tool(
            human_tool,
            {"action_uuid": str(uuid.uuid4()), "prompt": greeting_message},
            RunnableConfig(callbacks=self.callbacks),  # type: ignore
            max_retries=self.tool_call_max_retries,
        )
        state.messages.append(HumanMessage(content=str(tool_response.content)))

        return state

    async def end_interaction_node(
        self, state: StructuredReActGraphState
    ) -> StructuredReActGraphState:
        logger.info(f"{AGENT_LOG_PREFIX} Ending interaction")

        # Call farewell_user tool
        farewell_tool = self._get_tool(self.farewell_tool)
        if not farewell_tool:
            raise RuntimeError(f"{self.farewell_tool} tool not found")

        await self._call_tool(
            farewell_tool,
            {"action_uuid": str(uuid.uuid4())},
            RunnableConfig(callbacks=self.callbacks),  # type: ignore
            max_retries=self.tool_call_max_retries,
        )
        state.messages.append(AIMessage(content=self.farewell_message))
        state.end = True

        return state

    async def build_graph(self):
        await self._build_graph(state_schema=StructuredReActGraphState)
        logger.info("Agent Graph built and compiled successfully")
        return self.graph

    async def _build_graph(self, state_schema) -> CompiledStateGraph:
        logger.info("Building and compiling the Agent Graph")

        graph = StateGraph(state_schema=state_schema)
        graph.add_node("start_interaction", self.start_interaction_node)
        graph.add_node("agent", self.agent_node)
        graph.add_node("tool", self.tool_node)
        graph.add_node("end_interaction", self.end_interaction_node)
        graph.add_edge("start_interaction", "agent")
        graph.add_edge("tool", "agent")
        graph.add_edge("end_interaction", "__end__")
        conditional_edge_possible_outputs = {
            AgentDecision.TOOL: "tool",
            AgentDecision.END: "end_interaction",
        }
        graph.add_conditional_edges(
            "agent",
            self.conditional_edge,
            conditional_edge_possible_outputs,  # type: ignore[reportArgumentType]
        )

        graph.set_entry_point("start_interaction")
        self.graph = graph.compile()

        return self.graph

    async def conditional_edge(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, state: StructuredReActGraphState
    ) -> Literal[AgentDecision.END, AgentDecision.TOOL]:
        """Conditional edge to check the end flag."""

        return AgentDecision.END if state.end else AgentDecision.TOOL

    @staticmethod
    def validate_system_prompt(system_prompt: str) -> bool:
        """Validate system prompt."""

        errors = []
        if not system_prompt:
            errors.append("The system prompt cannot be empty.")

        required_prompt_variables = {
            "{tools}": "System prompt must contain {tools}.",
            "{tool_names}": "System prompt must contain {tool_names}.",
            "{structured_output}": "System prompt must contain {structured_output}.",
        }
        for variable_name, error_message in required_prompt_variables.items():
            if variable_name not in system_prompt:
                errors.append(error_message)
        if errors:
            error_text = "\n".join(errors)
            logger.exception("%s %s", AGENT_LOG_PREFIX, error_text)
            raise ValueError(error_text)
        return True
