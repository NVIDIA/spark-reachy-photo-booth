# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from typing import Any, Union, get_args, get_origin

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def get_item_type_schema(item_type: type[str | int | float | bool]) -> dict[str, Any]:
    """Get JSON schema for array item types"""
    if item_type is str:
        return {"type": "string"}
    elif item_type is int:
        return {"type": "integer"}
    elif item_type is float:
        return {"type": "number"}
    elif item_type is bool:
        return {"type": "boolean"}
    else:
        return {"type": "string"}


def get_parameter_schema(param_type: type, param_name: str) -> dict[str, Any]:
    """Get JSON schema for a parameter type"""
    if param_type is float:
        return {"type": "number", "description": f"Parameter {param_name}"}
    elif param_type is int:
        return {"type": "integer", "description": f"Parameter {param_name}"}
    elif param_type is str:
        return {"type": "string", "description": f"Parameter {param_name}"}
    elif param_type is bool:
        return {"type": "boolean", "description": f"Parameter {param_name}"}

    origin = get_origin(param_type)

    if origin is list:
        args = get_args(param_type)
        if args:
            item_type = args[0]
            item_schema = get_item_type_schema(item_type)
            return {
                "type": "array",
                "description": f"Parameter {param_name}",
                "items": item_schema,
            }
        else:
            return {
                "type": "array",
                "description": f"Parameter {param_name}",
                "items": {"type": "string"},
            }

    if origin is Union:
        args = get_args(param_type)
        non_none_types = [t for t in args if t is not type(None)]
        if len(non_none_types) == 1:
            return get_parameter_schema(non_none_types[0], param_name)

    if origin is dict:
        return {"type": "object", "description": f"Parameter {param_name}"}

    return {"type": "string", "description": f"Parameter {param_name}"}


def extract_and_parse_json(text: str) -> dict[str, Any]:
    """Extract and parse JSON from text, with multiple fallback strategies"""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        msg = (
            f"Failed to parse JSON after multiple cleanup attempts. "
            f"Original text: {text[:100]}..."
        )
        raise json.JSONDecodeError(msg, text, e.pos) from e


def convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Convert LangChain message to dict format"""
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        msg_dict = {"role": "assistant", "content": message.content or ""}
        if "tool_calls" in message.additional_kwargs:
            msg_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
        if "function_call" in message.additional_kwargs:
            msg_dict["function_call"] = message.additional_kwargs["function_call"]
        if "context" in message.additional_kwargs:
            msg_dict["context"] = message.additional_kwargs["context"]
        if "reasoning" in message.additional_kwargs:
            msg_dict["reasoning"] = message.additional_kwargs["reasoning"]
        return msg_dict
    elif isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, ToolMessage):
        msg_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
        if "name" in message.additional_kwargs:
            msg_dict["name"] = message.additional_kwargs["name"]
        return msg_dict
    elif isinstance(message, FunctionMessage):
        return {"role": "function", "content": message.content, "name": message.name}
    elif isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    else:
        return {"role": "user", "content": str(message.content)}


def convert_dict_to_message(msg_dict: dict[str, Any]) -> BaseMessage:
    """Convert dict to LangChain message"""
    role = msg_dict.get("role")
    if role == "user":
        return HumanMessage(content=msg_dict.get("content", ""))
    elif role == "assistant":
        content = msg_dict.get("content", "") or ""
        additional_kwargs: dict[str, Any] = {}
        if function_call := msg_dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        if tool_calls := msg_dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = tool_calls
        if context := msg_dict.get("context"):
            additional_kwargs["context"] = context
        if reasoning := msg_dict.get("reasoning"):
            additional_kwargs["reasoning"] = reasoning
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=msg_dict.get("content", ""))
    elif role == "function":
        return FunctionMessage(
            content=msg_dict.get("content", ""),
            name=msg_dict.get("name"),  # type: ignore[arg-type]
        )
    elif role == "tool":
        additional_kwargs = {}
        if "name" in msg_dict:
            additional_kwargs["name"] = msg_dict["name"]
        return ToolMessage(
            content=msg_dict.get("content", ""),
            tool_call_id=msg_dict.get("tool_call_id"),
            additional_kwargs=additional_kwargs,
        )
    else:
        return ChatMessage(content=msg_dict.get("content", ""), role=role)  # type: ignore[arg-type]


def convert_delta_to_message_chunk(
    delta: dict[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    """Convert a delta from streaming response to a message chunk"""
    role = delta.get("role")
    # Ensure content is always a string (can be None in streaming deltas)
    content = delta.get("content") or ""
    additional_kwargs: dict[str, Any] = {}

    if role == "assistant" or default_class == AIMessageChunk:
        if function_call := delta.get("function_call"):
            additional_kwargs["function_call"] = function_call
        if tool_calls := delta.get("tool_calls"):
            additional_kwargs["tool_calls"] = tool_calls
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)

    return default_class(content=content)  # type: ignore[call-arg]
