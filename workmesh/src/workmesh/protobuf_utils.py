# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for working with protobuf messages.

This module provides helper functions for converting between protobuf types
and native Python types, particularly for google.protobuf.Value objects.
"""

from collections.abc import Mapping
from typing import Any

from google.protobuf.struct_pb2 import NullValue, Value


def protobuf_value_to_python(value: Value) -> Any:
    """Convert a protobuf Value to a native Python value.

    This handles all protobuf Value types:
    - null_value -> None
    - number_value -> float
    - string_value -> str
    - bool_value -> bool
    - struct_value -> dict
    - list_value -> list

    Args:
        value: A google.protobuf.Value object

    Returns:
        The corresponding Python native type

    Example:
        >>> from google.protobuf.struct_pb2 import Value
        >>> value = Value()
        >>> value.string_value = "hello"
        >>> protobuf_value_to_python(value)
        'hello'
    """
    if not isinstance(value, Value):
        return value

    which = value.WhichOneof("kind")
    if which == "null_value":
        return None
    elif which == "number_value":
        return value.number_value
    elif which == "string_value":
        return value.string_value
    elif which == "bool_value":
        return value.bool_value
    elif which == "struct_value":
        return {
            k: protobuf_value_to_python(v) for k, v in value.struct_value.fields.items()
        }
    elif which == "list_value":
        return [protobuf_value_to_python(v) for v in value.list_value.values]
    return None


def protobuf_map_to_dict(value_map: Mapping[str, Value]) -> dict[str, Any]:
    """Convert a protobuf map<string, Value> to a Python dict.

    This is useful for unpacking the input field from ToolStatus messages.

    Args:
        value_map: A dictionary with string keys and protobuf Value objects as values

    Returns:
        A dictionary with string keys and native Python values

    Example:
        >>> from workmesh import ToolStatus
        >>> tool_status = ToolStatus(...)  # received from Kafka
        >>> input_dict = protobuf_map_to_dict(tool_status.input)
        >>> print(input_dict["prompt"])  # Access as normal Python dict
        'What is your name?'
    """
    return {key: protobuf_value_to_python(val) for key, val in value_map.items()}


def python_to_protobuf_value(py_value: Any) -> Value:
    """Convert a Python value to a protobuf Value.

    This handles all common Python types:
    - None -> null_value
    - bool -> bool_value
    - int/float -> number_value
    - str -> string_value
    - dict -> struct_value
    - list -> list_value

    Args:
        py_value: Any Python value

    Returns:
        A google.protobuf.Value object

    Example:
        >>> value = python_to_protobuf_value({"name": "Alice", "age": 30})
        >>> # Can now be assigned to a protobuf field expecting Value
    """
    value = Value()
    if py_value is None:
        value.null_value = NullValue.NULL_VALUE
    elif isinstance(py_value, bool):
        value.bool_value = py_value
    elif isinstance(py_value, (int, float)):
        value.number_value = float(py_value)
    elif isinstance(py_value, str):
        value.string_value = py_value
    elif isinstance(py_value, dict):
        for k, v in py_value.items():
            converted = python_to_protobuf_value(v)
            value.struct_value.fields[k].CopyFrom(converted)
    elif isinstance(py_value, list):
        for item in py_value:
            converted = python_to_protobuf_value(item)
            value.list_value.values.add().CopyFrom(converted)
    else:
        # Fallback: convert to string
        value.string_value = str(py_value)
    return value


def dict_to_protobuf_map(py_dict: dict[str, Any]) -> Mapping[str, Value]:
    """Convert a Python dict to a protobuf map<string, Value>.

    This is useful for populating the input field when creating ToolStatus messages.

    Args:
        py_dict: A Python dictionary with any values

    Returns:
        A dictionary with string keys and protobuf Value objects as values

    Example:
        >>> from workmesh import ToolStatus
        >>> input_data = {"prompt": "Hello", "count": 5}
        >>> tool_status = ToolStatus(
        ...     name="my_tool",
        ...     input=dict_to_protobuf_map(input_data),
        ... )
    """
    return {key: python_to_protobuf_value(val) for key, val in py_dict.items()}
