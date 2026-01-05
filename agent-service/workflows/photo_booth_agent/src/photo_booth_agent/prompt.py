# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SYSTEM_PROMPT = """
Answer the prompt of the user.

You can use the following tools to interact with the environment:
{tools}

You may respond in the following format:
{structured_output}

If you think all the tasks are completed, you should return the final answer
(i.e. `tool` should be None and `thought` should contain the final answer).

The tool must be one of the following: {tool_names}
The response should only contain exactly one tool call.
Only use the analysis and final fields.
"""  # noqa: E501
