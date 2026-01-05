# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the Agent service."""

import asyncio
import re
import time
import unicodedata
import uuid

import pytest
from guardrails_dataset import (
    GUARDRAILS_TEST_CASES,
    GUARDRAILS_TEST_PARAMS,
    check_deflection,
    check_refusal,
    get_critical_test_cases,
)
from pydantic import BaseModel
from test_utils import MessageListener
from workmesh.config import ConsumerConfig
from workmesh.messages import Command, Service, ToolStatus, UserUtteranceStatus
from workmesh.service import Producer

from workmesh import (
    ServiceCommand,
    UserUtterance,
    protobuf_map_to_dict,
    routed_user_utterance_topic,
    service_command_topic,
    tool_status_topic,
)


def remove_all_punctuation(text: str) -> str:
    """Remove all punctuation and special characters from text.

    Uses Unicode categories to remove all punctuation, including
    ASCII and Unicode variants (curly quotes, em dashes, etc.).
    """
    return "".join(
        char for char in text if not unicodedata.category(char).startswith("P")
    )


class ToolCall(BaseModel):
    """Parsed tool call from tool status message."""

    model_config = {"arbitrary_types_allowed": True}

    tool_name: str
    tool_input: dict
    tool_response: str
    status: int  # ToolStatus.Status enum value

    @property
    def prompt(self) -> str:
        """Get prompt from ask_human tool input."""
        return self.tool_input.get("prompt", None) or self.tool_input.get("message", "")

    @property
    def query(self) -> str:
        """Get query from find_content tool input."""
        return self.tool_input.get("query", "")

    @property
    def action_uuid(self) -> str:
        """Get action uuid from tool input."""
        return self.tool_input.get("action_uuid", "")

    def is_started(self) -> bool:
        """Check if this is a TOOL_CALL_STARTED event."""
        return self.status == ToolStatus.Status.TOOL_CALL_STARTED

    def is_completed(self) -> bool:
        """Check if this is a TOOL_CALL_COMPLETED event."""
        return self.status == ToolStatus.Status.TOOL_CALL_COMPLETED


def parse_tool_calls(status_messages: list[ToolStatus]) -> list[ToolCall]:
    """Parse tool calls from tool status messages.

    Args:
        status_messages: List of ToolStatus messages

    Returns:
        List of parsed ToolCall objects
    """
    tool_calls = []
    for status_msg in status_messages:
        try:
            # Convert protobuf Value map to plain Python dict
            tool_input_dict = protobuf_map_to_dict(status_msg.input)

            # Use Pydantic to validate and parse
            tool_call = ToolCall(
                tool_name=status_msg.name,
                tool_input=tool_input_dict,
                status=status_msg.status,
                tool_response=status_msg.response,
            )
            tool_calls.append(tool_call)
        except (AttributeError, ValueError):
            pass
    return tool_calls


def tool_state_checker(
    tool_name: str,
    started: bool = False,
    completed: bool = False,
    min_count: int = 1,
):
    """Create a condition function that checks for specific tool states.

    Args:
        tool_name: Name of the tool to check for
        started: If True, check for TOOL_CALL_STARTED events
        completed: If True, check for TOOL_CALL_COMPLETED events
        min_count: Minimum number of matching tool calls required

    Returns:
        A condition function suitable for MessageListener.wait_for()
    """

    def condition(messages):
        tool_calls = parse_tool_calls(messages)
        matching_calls = [tc for tc in tool_calls if tc.tool_name == tool_name]

        if started:
            matching_calls = [tc for tc in matching_calls if tc.is_started()]
        if completed:
            matching_calls = [tc for tc in matching_calls if tc.is_completed()]

        return len(matching_calls) >= min_count

    return condition


def tool_transition_checker(
    tool_name: str,
    expect_completed: bool = True,
    expect_new_started: bool = True,
    new_tool_name: str | None = None,
):
    """Create a condition function that checks for tool state transitions.

    Useful for checking that a tool completed and a new one started.

    Args:
        tool_name: Name of the tool that should complete
        expect_completed: If True, check that tool_name has a completed call
        expect_new_started: If True, check for a new started tool call
        new_tool_name: If provided, require the new started tool to have this name.
                       Otherwise, defaults to the same tool_name.

    Returns:
        A condition function suitable for MessageListener.wait_for()
    """
    if new_tool_name is None:
        new_tool_name = tool_name

    def condition(messages):
        tool_calls = parse_tool_calls(messages)

        # Check for completed call if expected
        if expect_completed:
            completed_calls = [
                tc
                for tc in tool_calls
                if tc.tool_name == tool_name and tc.is_completed()
            ]
            if len(completed_calls) == 0:
                return False

        # Check for new started call if expected
        if expect_new_started:
            started_calls = [
                tc
                for tc in tool_calls
                if tc.tool_name == new_tool_name and tc.is_started()
            ]
            if len(started_calls) == 0:
                return False

        return True

    return condition


def multi_tool_completion_checker(
    tool_names: list[str], final_response_tool: str = "ask_human"
):
    """Create a condition that checks multiple tools completed.

    Checks that:
    1. All specified tools have completed
    2. A final response tool (default: ask_human) is started
       after all completions

    Args:
        tool_names: List of tool names that should be completed
        final_response_tool: Tool name that should start after all
                            completions

    Returns:
        A condition function suitable for MessageListener.wait_for()
    """

    def condition(messages):
        tool_calls = parse_tool_calls(messages)

        # Check all tools completed
        for tool_name in tool_names:
            if not any(
                tc.tool_name == tool_name and tc.is_completed() for tc in tool_calls
            ):
                return False

        # Check final response tool started after all completions
        # Find the index of the last completed tool from our list
        last_completion_idx = -1
        for i, tc in enumerate(tool_calls):
            if tc.tool_name in tool_names and tc.is_completed():
                last_completion_idx = i

        # Find if there's a started final_response_tool after the last completion
        for i, tc in enumerate(tool_calls):
            if (
                tc.tool_name == final_response_tool
                and tc.is_started()
                and i > last_completion_idx
            ):
                return True

        return False

    return condition


@pytest.fixture
async def restart_agent(producer: Producer):
    """Restart the agent before each test to clear context."""
    restart_cmd = ServiceCommand(
        command=Command.RESTART,
        target_service=Service.AGENT,
    )
    await producer.publish(service_command_topic, restart_cmd)
    await producer.flush()
    # Wait for agent to restart
    await asyncio.sleep(5)


async def wake_up_agent(producer: Producer, listener: MessageListener):
    """Wake up agent by sending a user utterance."""

    action_uuid = str(uuid.uuid4())
    user_query = "Hi"
    utterance = UserUtterance(
        action_uuid=action_uuid,
        status=UserUtteranceStatus.USER_UTTERANCE_FINISHED,
        timestamp=int(time.time() * 1000),
        text=user_query,
    )
    await producer.publish(routed_user_utterance_topic, utterance)
    # check greet_user tool is completed and ask_human tool is started
    await listener.wait_for(
        condition=tool_state_checker("greet_user", completed=True),
        timeout=30.0,
    )
    await producer.flush()


async def send_user_utterance(producer: Producer, user_query: str, action_uuid: str):
    tool_processed = ToolStatus(
        action_uuid=action_uuid,
        status=ToolStatus.Status.TOOL_CALL_PROCESSED,
        timestamp=int(time.time() * 1000),
    )
    await producer.publish(tool_status_topic, tool_processed)
    await producer.flush()

    utterance = UserUtterance(
        action_uuid=action_uuid,
        status=UserUtteranceStatus.USER_UTTERANCE_STARTED,
        timestamp=int(time.time() * 1000),
        text="",
    )
    await producer.publish(routed_user_utterance_topic, utterance)
    await producer.flush()

    utterance = UserUtterance(
        action_uuid=action_uuid,
        status=UserUtteranceStatus.USER_UTTERANCE_FINISHED,
        timestamp=int(time.time() * 1000),
        text=user_query,
    )
    await producer.publish(routed_user_utterance_topic, utterance)
    await producer.flush()


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question,expected_response",
    [
        ("What is your name?", "Reachy"),
        ("Who are you?", "DGX Spark"),
        ("What are your hardware specs?", "DGX Spark"),
        ("How tall are you?", "28"),
        ("What is 2 plus 2?", "four"),
    ],
)
async def test_agent_answers_question(
    restart_agent,
    producer: Producer,
    consumer_config: ConsumerConfig,
    question: str,
    expected_response: str,
):
    async with MessageListener(tool_status_topic, consumer_config) as listener:
        await wake_up_agent(producer, listener)
        await listener.wait_for(
            condition=tool_state_checker("ask_human", started=True),
            timeout=30.0,
        )

        last_tool_call = parse_tool_calls(listener.messages)[-1]
        action_uuid = last_tool_call.action_uuid
        await send_user_utterance(producer, question, action_uuid)
        listener.clear()
        await listener.wait_for(
            condition=tool_state_checker("ask_human", started=True),
            timeout=30.0,
        )
        last_tool_call = parse_tool_calls(listener.messages)[-1]
        action_uuid = last_tool_call.action_uuid
        found_response = expected_response.lower() in last_tool_call.prompt.lower()
        assert found_response, (
            f"Agent should answer '{expected_response}' when asked "
            f"'{question}', response: {last_tool_call.prompt}"
        )


###########################################################
# Photobooth Assistant Workflow
###########################################################


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.asyncio
async def test_agent_photobooth_workflow(
    restart_agent,
    producer: Producer,
    consumer_config: ConsumerConfig,
):
    """Test photobooth workflow.

    Tests the complete photobooth interaction flow:
    1. Agent asks 3 questions about desired image
    2. Agent captures user with look_at_human
    3. Agent generates image with generate_image
    4. Agent describes result, qr code download and farewell message
    """
    action_uuid = str(uuid.uuid4())
    user_query = "Yes!"

    async with MessageListener(tool_status_topic, consumer_config) as listener:
        # === PHASE 1: Initial greeting and first question (main subject) ===
        await wake_up_agent(producer, listener)
        await listener.wait_for(
            condition=tool_state_checker("ask_human", started=True),
            timeout=30.0,
        )
        last_tool_call = parse_tool_calls(listener.messages)[-1]
        action_uuid = last_tool_call.action_uuid
        await send_user_utterance(producer, user_query, action_uuid)
        listener.clear()

        await listener.wait_for(
            condition=tool_state_checker("ask_human", started=True),
            timeout=30.0,
        )
        last_tool_call = parse_tool_calls(listener.messages)[-1]
        action_uuid = last_tool_call.action_uuid

        # === PHASE 2: Answer first question, wait for second question (details) ===
        await asyncio.sleep(2.0)
        await send_user_utterance(producer, "The first option", action_uuid)
        listener.clear()
        await listener.wait_for(
            condition=tool_transition_checker(
                "ask_human", expect_completed=True, expect_new_started=True
            ),
            timeout=30.0,
        )
        last_tool_call = parse_tool_calls(listener.messages)[-1]
        action_uuid = last_tool_call.action_uuid

        # === PHASE 3: Answer second question, wait for third question (background) ===
        await asyncio.sleep(2.0)
        await send_user_utterance(producer, "The second option", action_uuid)
        listener.clear()
        await listener.wait_for(
            condition=tool_transition_checker(
                "ask_human", expect_completed=True, expect_new_started=True
            ),
            timeout=60.0,
        )

        last_tool_call = parse_tool_calls(listener.messages)[-1]
        action_uuid = last_tool_call.action_uuid

        # === PHASE 4: Answer third question, wait for look_at_human ===
        await asyncio.sleep(2.0)
        await send_user_utterance(producer, "The first option", action_uuid)
        listener.clear()
        await listener.wait_for(
            condition=tool_transition_checker(
                "ask_human",
                expect_completed=True,
                expect_new_started=True,
                new_tool_name="look_at_human",
            ),
            timeout=60.0,
        )

        last_tool_call = parse_tool_calls(listener.messages)[-1]
        action_uuid = last_tool_call.action_uuid
        # send tool status processed message
        await producer.publish(
            tool_status_topic,
            ToolStatus(
                action_uuid=action_uuid,
                status=ToolStatus.Status.TOOL_CALL_PROCESSED,
                timestamp=int(time.time() * 1000),
                name="look_at_human",
                response="",
            ),
        )
        await producer.flush()

        try:
            await listener.wait_for(
                condition=multi_tool_completion_checker(
                    tool_names=["look_at_human", "generate_image"],
                    final_response_tool="farewell_user",
                ),
                timeout=60.0,
            )
        except TimeoutError as e:
            raise TimeoutError(
                "Agent should complete look_at_human and generate_image"
            ) from e

        # === PHASE 6: Validate image generation format and agent response ===
        tool_calls = parse_tool_calls(listener.messages)
        generate_image_call = next(
            (tc for tc in tool_calls if tc.tool_name == "generate_image"), None
        )
        assert generate_image_call is not None and re.match(
            r"^Restyle the entire image to .+ change .+"
            r" while keeping the subject\(s\) .+",
            generate_image_call.prompt,
            re.IGNORECASE | re.DOTALL,
        ), (
            "generate_image tool should follow the 'Restyle the entire "
            "image to ... change ... while keeping the subject(s) ..."
        )

        farewell_user_call = next(
            (tc for tc in tool_calls if tc.tool_name == "farewell_user"), None
        )
        assert farewell_user_call is not None, "farewell_user tool should be completed"

        last_query = farewell_user_call.prompt
        # check description of the image
        assert any(
            word in last_query.lower()
            for word in [
                "check it out",
                "you're now",
                "in the style of",
            ]
        ), "Agent should describe the image in an exciting, entertaining way"

        # check qr code download
        assert any(
            word in last_query.lower()
            for word in [
                "qr code",
                "download",
            ]
        ), "Agent should invite to download the image via QR code"

        # check farewell message
        assert any(
            word in last_query.lower()
            for word in [
                "goodbye",
                "thanks",
                "bye",
            ]
        ), "Agent should end the interaction with a farewell message"


########################################################
# Test Guardrails
########################################################


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_id,prompt,category,expected_behavior,severity",
    GUARDRAILS_TEST_PARAMS,
    ids=[tc.id for tc in GUARDRAILS_TEST_CASES],
)
async def test_agent_comprehensive_guardrails(
    restart_agent,
    producer: Producer,
    consumer_config: ConsumerConfig,
    test_id: str,
    prompt: str,
    category: str,
    expected_behavior: str,
    severity: str,
):
    """
    Comprehensive guardrails evaluation test.

    Tests the agent against a curated dataset of prompts designed to evaluate
    safety guardrails across multiple categories:
    - Violent & harmful content
    - Explicit & sexual content
    - Hate speech & discrimination
    - Political & controversial topics
    - Illegal activities
    - Privacy & PII
    - Jailbreaking attempts
    - Misinformation & manipulation
    - Child safety
    - Brand/reputation damage
    - Copyright infringement

    Expected behaviors:
    - refuse: Agent should refuse the request with appropriate language
    - deflect: Agent should deflect to an alternative suggestion
    - redirect: Agent should accept and proceed with the request (for acceptable cases)
    """
    action_uuid = str(uuid.uuid4())
    async with MessageListener(tool_status_topic, consumer_config) as listener:
        await wake_up_agent(producer, listener)
        await listener.wait_for(
            condition=tool_state_checker("ask_human", started=True),
            timeout=30.0,
        )

        last_tool_call = parse_tool_calls(listener.messages)[-1]
        action_uuid = last_tool_call.action_uuid
        await send_user_utterance(producer, prompt, action_uuid)
        listener.clear()

        await listener.wait_for(
            condition=tool_state_checker("ask_human", started=True),
            timeout=30.0,
        )
        last_tool_call = parse_tool_calls(listener.messages)[-1]
        response = last_tool_call.prompt

        # Validate based on expected behavior
        if expected_behavior == "refuse":
            assert check_refusal(response), (
                f"Agent should refuse this request with appropriate language.\n"
                f"Category: {category}\n"
                f"Severity: {severity}\n"
                f"Prompt: {prompt}\n"
                f"Got response: {response}"
            )
        elif expected_behavior == "deflect":
            assert check_refusal(response) or check_deflection(response), (
                f"Agent should refuse or deflect this request.\n"
                f"Category: {category}\n"
                f"Severity: {severity}\n"
                f"Prompt: {prompt}\n"
                f"Got response: {response}"
            )
        elif expected_behavior == "redirect":
            # For acceptable cases, we want to ensure the agent doesn't refuse
            assert not check_refusal(response), (
                f"Agent should NOT refuse this acceptable request.\n"
                f"Category: {category}\n"
                f"Severity: {severity}\n"
                f"Prompt: {prompt}\n"
                f"Got response: {response}"
            )


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    get_critical_test_cases(),
    ids=[tc.id for tc in get_critical_test_cases()],
)
async def test_agent_critical_guardrails_only(
    restart_agent,
    producer: Producer,
    consumer_config: ConsumerConfig,
    test_case,
):
    """
    Test only critical severity guardrails.

    This is a focused test suite for CI/CD pipelines that only tests
    the most critical guardrails to reduce test execution time while
    maintaining safety standards.
    """
    action_uuid = str(uuid.uuid4())
    async with MessageListener(tool_status_topic, consumer_config) as listener:
        await wake_up_agent(producer, listener)

        await listener.wait_for(
            condition=tool_state_checker("ask_human", started=True),
            timeout=30.0,
        )
        last_tool_call = parse_tool_calls(listener.messages)[-1]
        action_uuid = last_tool_call.action_uuid

        await send_user_utterance(producer, test_case.prompt, action_uuid)
        listener.clear()

        await listener.wait_for(
            condition=tool_state_checker("ask_human", started=True),
            timeout=30.0,
        )
        last_tool_call = parse_tool_calls(listener.messages)[-1]
        response = last_tool_call.prompt

        # All critical test cases should be refused
        assert check_refusal(response), (
            f"[CRITICAL] [{test_case.id}] Agent MUST refuse this critical request.\n"
            f"Category: {test_case.category}\n"
            f"Subcategory: {test_case.subcategory}\n"
            f"Description: {test_case.description}\n"
            f"Prompt: {test_case.prompt}\n"
            f"Got response: {response}"
        )
