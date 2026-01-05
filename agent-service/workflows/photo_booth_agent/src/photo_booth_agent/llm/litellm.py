# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import litellm
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from litellm.types.utils import ModelResponse
from nat.builder.builder import Builder, LLMFrameworkEnum
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_client, register_llm_provider
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.plugins.langchain.llm import patch_with_retry
from pydantic import BaseModel, ConfigDict, Field

from photo_booth_agent.constant import SERVICE_NAME
from photo_booth_agent.llm.utils import (
    convert_delta_to_message_chunk,
    convert_message_to_dict,
    extract_and_parse_json,
    get_parameter_schema,
)

logger = logging.getLogger(SERVICE_NAME)


class CompletionClientConfig(RetryMixin, LLMBaseConfig, name="completion_client"):
    """Configuration for Completion API (OpenAI-style) client."""

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    model_name: str = Field(description="The model name to use.")
    api_key: str | None = Field(
        default=None, description="API key to interact with hosted model."
    )
    base_url: str | None = Field(
        default=None, description="Base URL to the hosted model."
    )
    temperature: float = Field(
        default=0.7, description="Sampling temperature in [0, 2]."
    )
    max_tokens: int | None = Field(
        default=None, description="Maximum number of tokens to generate."
    )
    top_p: float = Field(default=1.0, description="Top-p for distribution sampling.")
    streaming: bool = Field(default=False, description="Whether to stream responses.")
    additional_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the API.",
    )


@dataclass
class ToolParameter:
    """Represents a single parameter in a tool function"""

    name: str
    param_type: type
    description: str
    required: bool
    default: Any = None


@dataclass
class Tool:
    """Standardized tool representation across all LLM APIs"""

    name: str
    description: str
    parameters: list[ToolParameter]
    callable: Callable


def create_tool_from_callable(tool_callable: Callable) -> Tool:
    """Extract Tool metadata from a Python function"""
    sig = inspect.signature(tool_callable)
    docstring = tool_callable.__doc__ or f"Call the {tool_callable.__name__} function"

    parameters = []
    for param_name, param in sig.parameters.items():
        param_type = (
            param.annotation if param.annotation != inspect.Parameter.empty else str
        )
        param_schema = get_parameter_schema(param_type, param_name)

        parameters.append(
            ToolParameter(
                name=param_name,
                param_type=param_type,
                description=param_schema.get("description", f"Parameter {param_name}"),
                required=param.default == inspect.Parameter.empty,
                default=param.default
                if param.default != inspect.Parameter.empty
                else None,
            )
        )

    return Tool(
        name=tool_callable.__name__,
        description=docstring,
        parameters=parameters,
        callable=tool_callable,
    )


def create_tool_from_basetool(base_tool: BaseTool) -> Tool:
    """Convert a LangChain BaseTool to our Tool format"""
    parameters = []
    for param_name, field_info in base_tool.input_schema.model_fields.items():
        param_type = field_info.annotation if field_info.annotation else str
        param_schema = get_parameter_schema(param_type, param_name)

        parameters.append(
            ToolParameter(
                name=param_name,
                param_type=param_type,
                description=field_info.description
                or param_schema.get("description", f"Parameter {param_name}"),
                required=field_info.is_required(),
                default=field_info.default if field_info.default is not None else None,
            )
        )

    return Tool(
        name=base_tool.name,
        description=base_tool.description or f"Call the {base_tool.name} function",
        parameters=parameters,
        callable=base_tool._run,
    )


@dataclass
class ToolCall:
    """Standardized tool call representation across all LLM APIs"""

    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    """Standardized response from any LLM API"""

    raw_response: Any
    additional_kwargs: dict[str, Any]
    content: str | BaseModel
    tool_calls: list[ToolCall]
    finish_reason: Literal["stop", "tool_calls", "length", "error"]
    assistant_message: dict[str, Any]


class UnifiedLLM(BaseChatModel, ABC):
    """
    Abstract base class that unifies different LLM APIs under a common interface.

    This class extends LangChain's BaseChatModel while providing a simplified
    call_llm_and_parse method for direct interaction with various LLM APIs.
    """

    model_name: str = Field(..., description="The model name to use.")
    api_key: str | None = Field(default=None, description="The API key to use.")
    base_url: str | None = Field(default=None, description="The base URL to use.")
    temperature: float = Field(default=0.7, description="The temperature to use.")
    max_tokens: int | None = Field(default=None, description="Max num. of tokens")
    additional_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the API.",
    )
    tools: list[BaseTool] = Field(default_factory=list, description="The tools to use.")
    output_model: type[BaseModel] | None = Field(
        default=None, description="The output model to use."
    )

    def bind_tools(  # type: ignore[override]
        self, tools: list[BaseTool]
    ) -> "UnifiedLLM":
        """Bind tools to this LLM instance"""
        self.tools = tools
        return self

    def with_structured_output(  # type: ignore[override]
        self, schema: type[BaseModel]
    ) -> "UnifiedLLM":
        """Enable structured output with a Pydantic model"""
        self.output_model = schema
        return self

    @abstractmethod
    def call_llm_and_parse(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        output_model: type[BaseModel] | None,
        invocation_params: dict[str, Any],
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """
        Single method that:
        1. Transforms messages to API-specific format (if needed)
        2. Calls the LLM API
        3. Extracts tool calls (if any) and returns early
        4. If no tool calls, parses structured output (if requested)
        5. Returns everything in standardized LLMResponse

        Raises:
        - ValidationError: if output_model validation fails
        - json.JSONDecodeError: if JSON parsing fails
        - Other exceptions for API errors
        """
        pass

    @abstractmethod
    async def acall_llm_and_parse(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        output_model: type[BaseModel] | None,
        invocation_params: dict[str, Any],
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """
        Async version of call_llm_and_parse.
        """
        pass

    @abstractmethod
    def call_llm_streaming(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        output_model: type[BaseModel] | None,
        invocation_params: dict[str, Any],
        stop: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Streaming version that yields chunks from the LLM API.
        """
        pass

    @abstractmethod
    async def acall_llm_streaming(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        output_model: type[BaseModel] | None,
        invocation_params: dict[str, Any],
        stop: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Async streaming version that yields chunks from the LLM API.
        """
        pass

    def _get_invocation_params(  # type: ignore[override]
        self, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Get parameters used for invocation (for tracing)

        Note: These parameters are non request specific.
        """

        params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "api_key": "***" if self.api_key else None,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
            **self.additional_kwargs,
            **kwargs,
        }

        return params

    @abstractmethod
    def _get_api_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        output_model: type[BaseModel] | None,
        invocation_params: dict[str, Any],
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get API parameters for the LLM API"""
        pass

    def _llm_response_to_chat_result(
        self, llm_response: LLMResponse, invocation_params: dict[str, Any]
    ) -> ChatResult:
        """Convert LLMResponse to LangChain ChatResult format"""
        if llm_response.tool_calls:
            # Build message with tool calls
            content = (
                llm_response.content if isinstance(llm_response.content, str) else ""
            )
            tool_calls_list = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in llm_response.tool_calls
            ]
            message = AIMessage(
                content=content,
                additional_kwargs={
                    "tool_calls": tool_calls_list,
                    **llm_response.additional_kwargs,
                },
            )
        else:
            content = llm_response.content
            if isinstance(content, BaseModel):
                content = content.model_dump_json()
            message = AIMessage(
                content=content, additional_kwargs=llm_response.additional_kwargs
            )

        generation = ChatGeneration(
            message=message,
            generation_info={
                "finish_reason": llm_response.finish_reason,
                "invocation_params": invocation_params,
                "additional_kwargs": llm_response.additional_kwargs,
            },
        )

        return ChatResult(generations=[generation])

    def _process_chunk(
        self,
        chunk_dict: dict[str, Any],
        default_chunk_class: type,
        first_chunk: bool,
        invocation_params: dict[str, Any],
    ) -> tuple[ChatGenerationChunk | None, type, bool]:
        """
        Process a single chunk from streaming API.

        Returns:
            Tuple of (ChatGenerationChunk or None, updated chunk class,
            updated first_chunk flag)
        """
        if not chunk_dict or "choices" not in chunk_dict:
            return None, default_chunk_class, first_chunk

        if len(chunk_dict["choices"]) == 0:
            return None, default_chunk_class, first_chunk

        choice = chunk_dict["choices"][0]
        delta = choice.get("delta")
        if delta is None:
            return None, default_chunk_class, first_chunk

        chunk = convert_delta_to_message_chunk(delta, default_chunk_class)
        finish_reason = choice.get("finish_reason")
        generation_info: dict[str, Any] = {}
        if finish_reason is not None:
            generation_info["finish_reason"] = finish_reason
        # Include invocation params in first chunk for tracing
        if first_chunk:
            generation_info["invocation_params"] = invocation_params
            first_chunk = False

        default_chunk_class = chunk.__class__
        cg_chunk = ChatGenerationChunk(
            message=chunk,
            generation_info=generation_info if generation_info else None,
        )
        return cg_chunk, default_chunk_class, first_chunk

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """LangChain interface: generate chat result from messages"""
        # Convert LangChain messages to dict format
        message_dicts = [convert_message_to_dict(msg) for msg in messages]

        # Convert LangChain tools to our Tool format
        converted_tools = [create_tool_from_basetool(tool) for tool in self.tools]

        # Capture invocation params for tracing
        invocation_params = self._get_invocation_params(stop=stop, **kwargs)

        # Call the unified API
        llm_response = self.call_llm_and_parse(
            messages=message_dicts,
            tools=converted_tools,
            output_model=self.output_model,
            stop=stop,
            invocation_params=invocation_params,
        )

        # Convert back to LangChain format
        return self._llm_response_to_chat_result(llm_response, invocation_params)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async LangChain interface: generate chat result from messages"""
        # Convert LangChain messages to dict format
        message_dicts = [convert_message_to_dict(msg) for msg in messages]

        # Convert LangChain tools to our Tool format
        converted_tools = [create_tool_from_basetool(tool) for tool in self.tools]

        # Capture invocation params for tracing
        invocation_params = self._get_invocation_params(stop=stop, **kwargs)

        # Call the unified async API
        llm_response = await self.acall_llm_and_parse(
            messages=message_dicts,
            tools=converted_tools,
            output_model=self.output_model,
            stop=stop,
            invocation_params=invocation_params,
            **kwargs,
        )

        # Convert back to LangChain format
        return self._llm_response_to_chat_result(llm_response, invocation_params)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat generation from messages"""
        # Convert LangChain messages to dict format
        message_dicts = [convert_message_to_dict(msg) for msg in messages]

        # Convert LangChain tools to our Tool format
        converted_tools = [create_tool_from_basetool(tool) for tool in self.tools]

        # Capture invocation params for tracing
        invocation_params = self._get_invocation_params(stop=stop, **kwargs)

        # Stream from the unified API
        default_chunk_class = AIMessageChunk
        first_chunk = True
        for chunk_dict in self.call_llm_streaming(
            messages=message_dicts,
            tools=converted_tools,
            output_model=self.output_model,
            stop=stop,
            invocation_params=invocation_params,
            **kwargs,
        ):
            cg_chunk, default_chunk_class, first_chunk = self._process_chunk(
                chunk_dict, default_chunk_class, first_chunk, invocation_params
            )
            if cg_chunk is None:
                continue

            if run_manager:
                run_manager.on_llm_new_token(cg_chunk.text, chunk=cg_chunk)
            yield cg_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream chat generation from messages"""
        # Convert LangChain messages to dict format
        message_dicts = [convert_message_to_dict(msg) for msg in messages]

        # Convert LangChain tools to our Tool format
        converted_tools = [create_tool_from_basetool(tool) for tool in self.tools]

        # Capture invocation params for tracing
        invocation_params = self._get_invocation_params(stop=stop, **kwargs)

        # Stream from the unified async API
        default_chunk_class = AIMessageChunk
        first_chunk = True
        stream: AsyncIterator[dict[str, Any]] = self.acall_llm_streaming(  # type: ignore (stream: True)
            messages=message_dicts,
            tools=converted_tools,
            output_model=self.output_model,
            stop=stop,
            invocation_params=invocation_params,
            **kwargs,
        )
        async for chunk_dict in stream:
            cg_chunk, default_chunk_class, first_chunk = self._process_chunk(
                chunk_dict, default_chunk_class, first_chunk, invocation_params
            )
            if cg_chunk is None:
                continue

            if run_manager:
                await run_manager.on_llm_new_token(cg_chunk.text, chunk=cg_chunk)
            yield cg_chunk

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type"""
        return "unified_llm"


class CompletionClient(UnifiedLLM):
    """
    Client for OpenAI-style Completion API.
    Uses standard message format without transformation.
    """

    model_name: str = Field(..., description="The model name to use.")
    api_key: str | None = Field(default=None, description="The API key to use.")
    base_url: str | None = Field(default=None, description="The base URL to use.")
    temperature: float = Field(default=0.7, description="The temperature to use.")
    max_tokens: int | None = Field(default=None, description="Max num. of tokens")
    additional_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the API.",
    )

    def _convert_tool_to_schema(self, tool: Tool) -> dict[str, Any]:
        """Convert Tool object to Completion API schema format"""
        properties = {}
        required = []

        for param in tool.parameters:
            param_schema = get_parameter_schema(param.param_type, param.name)
            properties[param.name] = param_schema

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def _prepare_messages(
        self, messages: list[dict[str, Any]], output_model: type[BaseModel] | None
    ) -> list[dict[str, Any]]:
        """Prepare messages with optional structured output instructions"""
        messages_copy = messages.copy()

        if output_model is not None:
            fields_desc = []
            for field_name, field_info in output_model.model_fields.items():
                field_type = str(field_info.annotation).replace("typing.", "")
                fields_desc.append(f'"{field_name}": {field_type}')

            fields_str = ", ".join(fields_desc)
            schema_instruction = (
                "\n\nIMPORTANT: After gathering all necessary information, "
                f"respond with ONLY a JSON object (no additional text) "
                f"with these fields: {{{fields_str}}}. "
                "Do not call functions if you already have the needed "
                "information."
            )
            if not any(msg.get("role") == "system" for msg in messages_copy):
                messages_copy.insert(
                    0, {"role": "system", "content": schema_instruction}
                )
            else:
                for idx, msg in enumerate(messages_copy):
                    if msg.get("role") == "system":
                        messages_copy[idx] = msg.copy()
                        messages_copy[idx]["content"] = (
                            msg["content"] + schema_instruction
                        )
                        break

        return messages_copy

    def _get_api_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        output_model: type[BaseModel] | None,
        invocation_params: dict[str, Any],
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get API parameters for the LLM API"""

        messages_copy = self._prepare_messages(messages, output_model)
        api_params = invocation_params.copy()
        api_params["stop"] = stop
        api_params["messages"] = messages_copy
        if tools:
            api_params["tools"] = [self._convert_tool_to_schema(tool) for tool in tools]
        if output_model is not None:
            api_params["response_format"] = {"type": "json_object"}
        return api_params

    def _parse_response(
        self,
        raw_response: ModelResponse,
        output_model: type[BaseModel] | None,
    ) -> LLMResponse:
        """Parse the response from the LLM API"""
        choices: list[litellm.Choices] = raw_response.choices  # type: ignore
        raw_tool_calls = choices[0].message.tool_calls
        additional_kwargs: dict[str, Any] = (
            choices[0].message.provider_specific_fields or {}
        )
        usage: litellm.Usage = raw_response.get("usage") or litellm.Usage(
            completion_tokens=None, prompt_tokens=None, total_tokens=None
        )
        finish_reason = None
        if (
            usage
            and self.max_tokens is not None
            and usage.completion_tokens == self.max_tokens
        ):
            finish_reason = "length"

        if raw_tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name or "unknown",
                    arguments=tc.function.arguments,
                )
                for tc in raw_tool_calls
            ]

            return LLMResponse(
                raw_response=raw_response,
                content="",
                tool_calls=tool_calls,
                finish_reason=finish_reason or "tool_calls",
                assistant_message={
                    "role": "assistant",
                    "content": raw_response.choices[0].message.content,  # type: ignore
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in raw_tool_calls
                    ],
                },
                additional_kwargs={**additional_kwargs, "finish_reason": finish_reason},
            )

        text_content = raw_response.choices[0].message.content or ""  # type: ignore

        if output_model:
            json_data = extract_and_parse_json(text_content)
            parsed_content = output_model(**json_data)

            return LLMResponse(
                raw_response=raw_response,
                content=parsed_content,
                tool_calls=[],
                finish_reason=finish_reason or "stop",
                assistant_message={"role": "assistant", "content": text_content},
                additional_kwargs={**additional_kwargs, "finish_reason": finish_reason},
            )

        return LLMResponse(
            raw_response=raw_response,
            content=text_content,
            tool_calls=[],
            finish_reason=finish_reason or "stop",
            assistant_message={"role": "assistant", "content": text_content},
            additional_kwargs={**additional_kwargs, "finish_reason": finish_reason},
        )

    def call_llm_and_parse(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        output_model: type[BaseModel] | None,
        invocation_params: dict[str, Any],
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """
        Completion API uses standard message format, so no transformation needed.
        Messages are passed directly to the API.
        """

        api_params = self._get_api_params(
            messages, tools, output_model, invocation_params, stop
        )

        raw_response: litellm.ModelResponse = litellm.completion(**api_params)  # type: ignore
        return self._parse_response(raw_response, output_model)

    async def acall_llm_and_parse(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        output_model: type[BaseModel] | None,
        invocation_params: dict[str, Any],
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Async version of call_llm_and_parse"""

        api_params = self._get_api_params(
            messages, tools, output_model, invocation_params, stop
        )

        logger.info(f"Calling async completion API with model: {self.model_name}")
        raw_response: litellm.ModelResponse = await litellm.acompletion(**api_params)  # type: ignore
        return self._parse_response(raw_response, output_model)

    def call_llm_streaming(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        output_model: type[BaseModel] | None,
        invocation_params: dict[str, Any],
        stop: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Streaming version that yields chunks"""
        api_params = self._get_api_params(
            messages, tools, output_model, invocation_params, stop
        )
        api_params["stream"] = True

        logger.info(f"Calling streaming completion API with model: {self.model_name}")

        stream: litellm.CustomStreamWrapper = litellm.completion(**api_params)  # type: ignore (stream: True)
        for chunk in stream:
            if chunk is None:
                continue
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            yield chunk

    async def acall_llm_streaming(  # type: ignore[override]
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        output_model: type[BaseModel] | None,
        invocation_params: dict[str, Any],
        stop: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async streaming version that yields chunks"""
        api_params = self._get_api_params(
            messages, tools, output_model, invocation_params, stop
        )
        api_params["stream"] = True

        logger.info(
            f"Calling async streaming completion API with model: {self.model_name}"
        )
        stream: litellm.CustomStreamWrapper = await litellm.acompletion(**api_params)  # type: ignore (stream: True)
        async for chunk in stream:
            if chunk is None:
                continue
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            yield chunk

    @property
    def _llm_type(self) -> str:
        return "completion_client"


@register_llm_provider(config_type=CompletionClientConfig)
async def completion_client_provider(config: CompletionClientConfig, builder: Builder):
    """Register CompletionClient as an LLM provider."""
    yield LLMProviderInfo(
        config=config,
        description="A unified Completion API client for OpenAI-style models.",
    )


@register_llm_client(
    config_type=CompletionClientConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN
)
async def completion_client_langchain(
    llm_config: CompletionClientConfig, builder: Builder
):
    """Create a CompletionClient instance for LangChain."""

    kwargs = {
        "model_name": llm_config.model_name,
        "api_key": llm_config.api_key,
        "base_url": llm_config.base_url,
        "temperature": llm_config.temperature,
        "max_tokens": llm_config.max_tokens,
        "additional_kwargs": llm_config.additional_kwargs,
    }

    client = CompletionClient(**kwargs)
    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(
            client,
            retries=llm_config.num_retries,
            retry_codes=llm_config.retry_on_status_codes,
            retry_on_messages=llm_config.retry_on_errors,
        )

    yield client
