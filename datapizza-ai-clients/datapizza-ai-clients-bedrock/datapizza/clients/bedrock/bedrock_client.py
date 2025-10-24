import os
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

import aioboto3
import boto3
from botocore.config import Config
from datapizza.core.cache import Cache
from datapizza.core.clients import Client, ClientResponse
from datapizza.memory import Memory
from datapizza.tools import Tool
from datapizza.type import FunctionCallBlock, TextBlock

from .memory_adapter import BedrockMemoryAdapter


class BedrockClient(Client):
    """A client for interacting with AWS Bedrock API.

    This class provides methods for invoking AWS Bedrock models (Claude, etc.)
    to generate responses. It extends the Client class and supports authentication
    via AWS profile or access keys.
    """

    def __init__(
        self,
        model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        system_prompt: str = "",
        temperature: float | None = None,
        cache: Cache | None = None,
        region_name: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        profile_name: str | None = None,
    ):
        """
        Args:
            model: The Bedrock model to use (e.g., 'anthropic.claude-3-5-sonnet-20241022-v2:0').
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for generation (0-1 for most models).
            cache: The cache to use for responses.
            region_name: AWS region name (default: us-east-1).
            aws_access_key_id: AWS access key ID (optional, can use AWS_ACCESS_KEY_ID env var).
            aws_secret_access_key: AWS secret access key (optional, can use AWS_SECRET_ACCESS_KEY env var).
            aws_session_token: AWS session token (optional, for temporary credentials).
            profile_name: AWS profile name (optional, can use AWS_PROFILE env var).
        """
        if temperature and not 0 <= temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1 for Bedrock models")

        super().__init__(
            model_name=model,
            system_prompt=system_prompt,
            temperature=temperature,
            cache=cache,
        )

        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        self.aws_session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")
        self.profile_name = profile_name or os.getenv("AWS_PROFILE")

        self.memory_adapter = BedrockMemoryAdapter()
        self._set_client()

    def _set_client(self):
        if not self.client:
            session_kwargs = {}

            # Priority: explicit credentials > profile > default credentials
            if self.aws_access_key_id and self.aws_secret_access_key:
                session_kwargs["aws_access_key_id"] = self.aws_access_key_id
                session_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
                if self.aws_session_token:
                    session_kwargs["aws_session_token"] = self.aws_session_token
            elif self.profile_name:
                session_kwargs["profile_name"] = self.profile_name

            session = boto3.Session(**session_kwargs)

            # Create bedrock-runtime client with retry configuration
            config = Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                read_timeout=300,
            )

            self.client = session.client(
                service_name="bedrock-runtime",
                region_name=self.region_name,
                config=config,
            )

    def _set_a_client(self):
        """Initialize async bedrock-runtime client using aioboto3"""
        if not self.a_client:
            session_kwargs = {}

            # Priority: explicit credentials > profile > default credentials
            if self.aws_access_key_id and self.aws_secret_access_key:
                session_kwargs["aws_access_key_id"] = self.aws_access_key_id
                session_kwargs["aws_secret_access_key"] = (
                    self.aws_secret_access_key
                )
                if self.aws_session_token:
                    session_kwargs["aws_session_token"] = self.aws_session_token
            elif self.profile_name:
                session_kwargs["profile_name"] = self.profile_name

            # Create async session
            session = aioboto3.Session(**session_kwargs)

            # Create bedrock-runtime client with retry configuration
            config = Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                read_timeout=300,
            )

            # Store the session and config for async client creation
            self.a_session = session
            self.a_config = config
            self.a_region_name = self.region_name

    def _convert_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """Convert tools to Bedrock tool format (similar to Anthropic)"""
        bedrock_tools = []
        for tool in tools:
            bedrock_tool = {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": tool.properties,
                            "required": tool.required,
                        }
                    },
                }
            }
            bedrock_tools.append(bedrock_tool)
        return bedrock_tools

    def _convert_tool_choice(
        self, tool_choice: Literal["auto", "required", "none"] | list[str]
    ) -> dict:
        """Convert tool choice to Bedrock format"""
        if isinstance(tool_choice, list):
            if len(tool_choice) > 1:
                raise NotImplementedError(
                    "Multiple function names not supported by Bedrock"
                )
            return {"tool": {"name": tool_choice[0]}}
        elif tool_choice == "required":
            return {"any": {}}
        elif tool_choice == "auto":
            return {"auto": {}}
        else:  # none
            return {}

    def _response_to_client_response(
        self, response: dict, tool_map: dict[str, Tool] | None = None
    ) -> ClientResponse:
        """Convert Bedrock response to ClientResponse"""
        blocks = []

        # Parse the response body
        response_body = response.get("output", {})

        # Handle message content
        message = response_body.get("message", {})
        content_items = message.get("content", [])

        for content_item in content_items:
            if "text" in content_item:
                blocks.append(TextBlock(content=content_item["text"]))
            elif "toolUse" in content_item:
                tool_use = content_item["toolUse"]
                tool = tool_map.get(tool_use["name"]) if tool_map else None
                if not tool:
                    raise ValueError(f"Tool {tool_use['name']} not found")

                blocks.append(
                    FunctionCallBlock(
                        id=tool_use["toolUseId"],
                        name=tool_use["name"],
                        arguments=tool_use["input"],
                        tool=tool,
                    )
                )

        # Extract usage information
        usage = response.get("usage", {})
        prompt_tokens = usage.get("inputTokens", 0)
        completion_tokens = usage.get("outputTokens", 0)

        # Extract stop reason
        stop_reason = response.get("stopReason")

        return ClientResponse(
            content=blocks,
            stop_reason=stop_reason,
            prompt_tokens_used=prompt_tokens,
            completion_tokens_used=completion_tokens,
            cached_tokens_used=0,  # Bedrock doesn't expose cache metrics in the same way
        )

    def _invoke(
        self,
        *,
        input: str,
        tools: list[Tool] | None,
        memory: Memory | None,
        tool_choice: Literal["auto", "required", "none"] | list[str],
        temperature: float | None,
        max_tokens: int,
        system_prompt: str | None,
        **kwargs,
    ) -> ClientResponse:
        """Implementation of the abstract _invoke method for Bedrock"""
        if tools is None:
            tools = []

        client = self._get_client()
        messages = self._memory_to_contents(None, input, memory)

        # Remove model role messages (Bedrock doesn't support this)
        messages = [message for message in messages if message.get("role") != "model"]

        tool_map = {tool.name: tool for tool in tools}

        # Build the request body according to Bedrock Converse API
        request_body = {
            "modelId": self.model_name,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens or 2048,
            },
        }

        if temperature is not None:
            request_body["inferenceConfig"]["temperature"] = temperature

        # Add system prompt if provided
        if system_prompt:
            request_body["system"] = [{"text": system_prompt}]

        # Add tools if provided
        if tools:
            request_body["toolConfig"] = {
                "tools": self._convert_tools(tools),
                "toolChoice": self._convert_tool_choice(tool_choice),
            }

        # Add any additional kwargs
        request_body.update(kwargs)

        # Call Bedrock Converse API
        response = client.converse(**request_body)

        return self._response_to_client_response(response, tool_map)

    async def _a_invoke(
        self,
        *,
        input: str,
        tools: list[Tool] | None,
        memory: Memory | None,
        tool_choice: Literal["auto", "required", "none"] | list[str],
        temperature: float | None,
        max_tokens: int,
        system_prompt: str | None,
        **kwargs,
    ) -> ClientResponse:
        """Async implementation using aioboto3"""
        if tools is None:
            tools = []

        # Ensure async client is initialized
        if not hasattr(self, "a_session"):
            self._set_a_client()

        messages = self._memory_to_contents(None, input, memory)

        # Remove model role messages (Bedrock doesn't support this)
        messages = [
            message for message in messages if message.get("role") != "model"
        ]

        tool_map = {tool.name: tool for tool in tools}

        # Build the request body according to Bedrock Converse API
        request_body = {
            "modelId": self.model_name,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens or 2048,
            },
        }

        if temperature is not None:
            request_body["inferenceConfig"]["temperature"] = temperature

        # Add system prompt if provided
        if system_prompt:
            request_body["system"] = [{"text": system_prompt}]

        # Add tools if provided
        if tools:
            request_body["toolConfig"] = {
                "tools": self._convert_tools(tools),
                "toolChoice": self._convert_tool_choice(tool_choice),
            }

        # Add any additional kwargs
        request_body.update(kwargs)

        # Call Bedrock Converse API asynchronously
        async with self.a_session.client(
            service_name="bedrock-runtime",
            region_name=self.a_region_name,
            config=self.a_config,
        ) as client:
            response = await client.converse(**request_body)

        return self._response_to_client_response(response, tool_map)

    def _stream_invoke(
        self,
        input: str,
        tools: list[Tool] | None,
        memory: Memory | None,
        tool_choice: Literal["auto", "required", "none"] | list[str],
        temperature: float | None,
        max_tokens: int,
        system_prompt: str | None,
        **kwargs,
    ) -> Iterator[ClientResponse]:
        """Implementation of streaming for Bedrock"""
        if tools is None:
            tools = []

        client = self._get_client()
        messages = self._memory_to_contents(None, input, memory)

        # Remove model role messages
        messages = [message for message in messages if message.get("role") != "model"]

        # Build the request body
        request_body = {
            "modelId": self.model_name,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens or 2048,
            },
        }

        if temperature is not None:
            request_body["inferenceConfig"]["temperature"] = temperature

        if system_prompt:
            request_body["system"] = [{"text": system_prompt}]

        if tools:
            request_body["toolConfig"] = {
                "tools": self._convert_tools(tools),
                "toolChoice": self._convert_tool_choice(tool_choice),
            }

        request_body.update(kwargs)

        # Call Bedrock ConverseStream API
        response = client.converse_stream(**request_body)

        # Process streaming response
        message_text = ""
        input_tokens = 0
        output_tokens = 0
        stop_reason = None

        stream = response.get("stream")
        if stream:
            for event in stream:
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        text_delta = delta["text"]
                        message_text += text_delta
                        yield ClientResponse(
                            content=[TextBlock(content=message_text)],
                            delta=text_delta,
                            stop_reason=None,
                            prompt_tokens_used=0,
                            completion_tokens_used=0,
                            cached_tokens_used=0,
                        )

                elif "metadata" in event:
                    metadata = event["metadata"]
                    usage = metadata.get("usage", {})
                    input_tokens = usage.get("inputTokens", 0)
                    output_tokens = usage.get("outputTokens", 0)

                elif "messageStop" in event:
                    stop_reason = event["messageStop"].get("stopReason")

        # Final response with complete information
        yield ClientResponse(
            content=[TextBlock(content=message_text)],
            delta="",
            stop_reason=stop_reason,
            prompt_tokens_used=input_tokens,
            completion_tokens_used=output_tokens,
            cached_tokens_used=0,
        )

    async def _a_stream_invoke(
        self,
        input: str,
        tools: list[Tool] | None,
        memory: Memory | None,
        tool_choice: Literal["auto", "required", "none"] | list[str],
        temperature: float | None,
        max_tokens: int,
        system_prompt: str | None,
        **kwargs,
    ) -> AsyncIterator[ClientResponse]:
        """Async streaming implementation using aioboto3"""
        if tools is None:
            tools = []

        # Ensure async client is initialized
        if not hasattr(self, "a_session"):
            self._set_a_client()

        messages = self._memory_to_contents(None, input, memory)

        # Remove model role messages
        messages = [
            message for message in messages if message.get("role") != "model"
        ]

        # Build the request body
        request_body = {
            "modelId": self.model_name,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens or 2048,
            },
        }

        if temperature is not None:
            request_body["inferenceConfig"]["temperature"] = temperature

        if system_prompt:
            request_body["system"] = [{"text": system_prompt}]

        if tools:
            request_body["toolConfig"] = {
                "tools": self._convert_tools(tools),
                "toolChoice": self._convert_tool_choice(tool_choice),
            }

        request_body.update(kwargs)

        # Call Bedrock ConverseStream API asynchronously
        async with self.a_session.client(
            service_name="bedrock-runtime",
            region_name=self.a_region_name,
            config=self.a_config,
        ) as client:
            response = await client.converse_stream(**request_body)

            # Process streaming response
            message_text = ""
            input_tokens = 0
            output_tokens = 0
            stop_reason = None

            stream = response.get("stream")
            if stream:
                async for event in stream:
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        if "text" in delta:
                            text_delta = delta["text"]
                            message_text += text_delta
                            yield ClientResponse(
                                content=[TextBlock(content=message_text)],
                                delta=text_delta,
                                stop_reason=None,
                                prompt_tokens_used=0,
                                completion_tokens_used=0,
                                cached_tokens_used=0,
                            )

                    elif "metadata" in event:
                        metadata = event["metadata"]
                        usage = metadata.get("usage", {})
                        input_tokens = usage.get("inputTokens", 0)
                        output_tokens = usage.get("outputTokens", 0)

                    elif "messageStop" in event:
                        stop_reason = event["messageStop"].get("stopReason")

            # Final response with complete information
            yield ClientResponse(
                content=[TextBlock(content=message_text)],
                delta="",
                stop_reason=stop_reason,
                prompt_tokens_used=input_tokens,
                completion_tokens_used=output_tokens,
                cached_tokens_used=0,
            )

    def _structured_response(
        self,
        input: str,
        output_cls: type,
        memory: Memory | None,
        temperature: float | None,
        max_tokens: int,
        system_prompt: str | None,
        tools: list[Tool] | None,
        tool_choice: Literal["auto", "required", "none"] | list[str] = "auto",
        **kwargs,
    ) -> ClientResponse:
        """Bedrock doesn't natively support structured output like OpenAI

        This would require prompting techniques or using Anthropic's prompt caching
        """
        raise NotImplementedError(
            "Bedrock doesn't natively support structured responses. "
            "Consider using prompt engineering or JSON mode with validation."
        )

    async def _a_structured_response(self, *args, **kwargs):
        raise NotImplementedError(
            "Bedrock doesn't natively support structured responses"
        )
