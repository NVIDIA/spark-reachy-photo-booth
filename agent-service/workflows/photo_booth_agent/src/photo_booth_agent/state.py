# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from photo_booth_agent.output import StructuredAgentAction


class StructuredReActGraphState(BaseModel):
    """
    State schema for the ReAct Agent Graph.

    Architecture:
    - messages: list[BaseMessage] - All history lives here:
      - User inputs (HumanMessage)
      - Agent thoughts (AIMessage with JSON)
      - Tool responses (ToolMessage)
      Note: they are persistent across sessions
    - agent_scratchpad: list[StructuredAgentAction] - Only holds current action
      Note: they are ephemeral and cleared after each tool call
    - end: bool - Simple flag to control graph routing
      Note: it is used to control the graph routing
    """

    messages: list[BaseMessage] = Field(default_factory=list)
    agent_scratchpad: list[StructuredAgentAction] = Field(default_factory=list)
    end: bool = Field(default=False)
