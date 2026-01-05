# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import logging
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, Field, field_validator

from photo_booth_agent.constant import SERVICE_NAME

logger = logging.getLogger(SERVICE_NAME)


def type_name(t):
    """Return a short, readable type name (handles Union, list, dict, etc.)."""
    origin = get_origin(t)
    if origin is Union:
        args = [type_name(a) for a in get_args(t)]
        return " | ".join(args)
    elif origin in (list, dict, tuple, set):
        args = [type_name(a) for a in get_args(t)] or ["Any"]
        return f"{origin.__name__}[{', '.join(args)}]"
    elif hasattr(t, "__name__"):
        return t.__name__
    return str(t)


class StructuredAgentAction(BaseModel):
    thought: str = Field(
        description="The thought of the agent. "
        "Should contain the final answer if the agent wants to return the final answer."
    )
    tool: str | None = Field(
        default=None,
        description="The tool to use. It can be `null` or empty if the agent wants "
        "to return the final answer.",
    )
    tool_input: dict[str, Any] = Field(
        default_factory=dict,
        description="The input to the tool. It can be empty if the agent wants "
        "to return the final answer.",
    )
    final_answer: str | None = Field(
        default=None,
        description="The final answer to the user's question.\n"
        "It can be empty if the agent wants to call a tool.",
    )

    @classmethod
    def simple_schema(cls) -> dict:
        """Returns a simple example format instead of verbose JSON schema"""

        return {
            f"{field}": (prop.description or "")
            + " The type of the field is "
            + type_name(prop.annotation)
            for field, prop in cls.model_fields.items()
        }

    @property
    def log(self) -> str:
        return json.dumps(self.model_dump())

    @field_validator("tool", mode="before")
    def validate_tool(cls, v: str | None) -> str | None:
        if isinstance(v, str):
            normalized = v.strip().lower()
            if normalized in ("", "none", "null"):
                return None
        if v is None:
            return None
        return v.strip()

    @field_validator("tool_input", mode="before")
    def validate_tool_input(cls, v: Any) -> dict[str, Any]:
        if isinstance(v, dict):
            return v

        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

            try:
                parsed = ast.literal_eval(v)  # pyright: ignore[reportArgumentType]
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                pass

        logger.warning(f"Could not parse tool_input as dict: {v!r}, using empty dict")
        return {}
