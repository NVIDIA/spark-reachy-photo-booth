# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from workmesh.messages import Robot, ToolStatus
from workmesh.service import Producer

from workmesh import dict_to_protobuf_map, tool_status_topic


class KafkaAsyncCallbackHandler(BaseCallbackHandler):
    """Async callback handler for LangChain."""

    def __init__(self, robot_id: Robot):
        self.robot_id = robot_id
        self.action_uuid_per_tool_call: dict[UUID, dict[str, Any]] = {}

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when the tool starts running.

        Args:
            serialized (dict[str, Any]): The serialized tool.
            input_str (str): The input string.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            metadata (Optional[dict[str, Any]]): The metadata.
            inputs (Optional[dict[str, Any]]): The inputs.
            kwargs (Any): Additional keyword arguments.
        """

        if isinstance(input_str, dict):
            tool_input_dict = input_str
        else:
            try:
                tool_input_dict = json.loads(input_str)
            except json.JSONDecodeError:
                tool_input_dict = ast.literal_eval(input_str)

        action_uuid = tool_input_dict.get("action_uuid")

        self.action_uuid_per_tool_call[run_id] = {
            "action_uuid": str(action_uuid),
            "tool_name": serialized["name"],
            "tool_input": tool_input_dict,
            "tool_response": "",
        }
        async with Producer() as producer:
            await producer.publish(
                tool_status_topic,
                ToolStatus(
                    robot_id=self.robot_id,
                    action_uuid=self.action_uuid_per_tool_call[run_id]["action_uuid"],
                    timestamp=int(time.time()),
                    status=ToolStatus.Status.TOOL_CALL_STARTED,
                    name=serialized["name"],
                    input=dict_to_protobuf_map(tool_input_dict),
                ),
            )

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when the tool ends running.

        Args:
            output (Any): The output of the tool.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """
        async with Producer() as producer:
            action_uuid = self.action_uuid_per_tool_call[run_id]["action_uuid"]

            tool_status = ToolStatus(
                robot_id=self.robot_id,
                action_uuid=action_uuid,
                timestamp=int(time.time()),
                status=ToolStatus.Status.TOOL_CALL_COMPLETED,
                name=self.action_uuid_per_tool_call[run_id]["tool_name"],
                input=dict_to_protobuf_map(
                    self.action_uuid_per_tool_call[run_id]["tool_input"]
                ),
                response=str(output),
            )
            del self.action_uuid_per_tool_call[run_id]

            await producer.publish(tool_status_topic, tool_status)

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors.

        Args:
            error (BaseException): The error that occurred.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """
        async with Producer() as producer:
            action_uuid = self.action_uuid_per_tool_call[run_id]["action_uuid"]

            tool_status = ToolStatus(
                robot_id=self.robot_id,
                action_uuid=action_uuid,
                timestamp=int(time.time()),
                status=ToolStatus.Status.TOOL_CALL_FAILED,
                name=self.action_uuid_per_tool_call[run_id]["tool_name"],
                input=dict_to_protobuf_map(
                    self.action_uuid_per_tool_call[run_id]["tool_input"]
                ),
                response=str(error),
            )
            del self.action_uuid_per_tool_call[run_id]

            await producer.publish(tool_status_topic, tool_status)
