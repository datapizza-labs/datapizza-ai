from collections.abc import AsyncIterator, Iterator
from typing import Literal

from datapizza.core.cache import Cache
from datapizza.core.clients import Client, ClientResponse
from datapizza.core.clients.models import TokenUsage
from datapizza.memory import Memory
from datapizza.tools import Tool
from datapizza.type import (
    FunctionCallBlock,
    Model,
    StructuredBlock,
    TextBlock,
    ThoughtBlock,
)

from google import genai
from google.genai import types
from google.oauth2 import service_account

from .memory_adapter import GoogleMemoryAdapter


class GoogleClient(Client):
    """A client for interacting with Google's Generative AI APIs.

    This class provides methods for invoking the Google GenAI API to generate responses
    based on given input data. It extends the Client class.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
        system_prompt: str = "",
        temperature: float | None = None,
        cache: Cache | None = None,
        project_id: str | None = None,
        location: str | None = None,
        credentials_path: str | None = None,
        use_vertexai: bool = False,
    ):
        """
        Args:
            api_key: The API key for the Google API.
            model: The model to use for the Google API.
            system_prompt: The system prompt to use for the Google API.
            temperature: The temperature to use for the Google API.
            cache: The cache to use for the Google API.
            project_id: The project ID for the Google API.
            location: The location for the Google API.
            credentials_path: The path to the credentials for the Google API.
            use_vertexai: Whether to use Vertex AI for the Google API.
        """
        if temperature and not 0 <= temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")

        super().__init__(
            model_name=model,
            system_prompt=system_prompt,
            temperature=temperature,
            cache=cache,
        )
        self.memory_adapter = GoogleMemoryAdapter()

        try:
            if use_vertexai:
                if not credentials_path:
                    raise ValueError("credentials_path must be provided")
                if not project_id:
                    raise ValueError("project_id must be provided")
                if not location:
                    raise ValueError("location must be provided")

                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                self.client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=location,
                    credentials=credentials,
                )
            else:
                if not api_key:
                    raise ValueError("api_key must be provided")

                self.client = genai.Client(api_key=api_key)

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Google GenAI client: {e!s}"
            ) from None

    def _convert_tool(self, tool: Tool) -> dict:
        """Convert tools to Google function format"""
        parameters = {
            "type": tool.schema["parameters"]["type"],
            "properties": tool.schema["parameters"]["properties"],
            "required": tool.schema["parameters"]["required"],
        }

        return {
            "name": tool.schema["name"],
            "description": tool.schema["description"],
            "parameters": parameters,
        }

    def _prepare_tools(self, tools: list[Tool] | None) -> list[types.Tool] | None:
        if not tools:
            return None

        google_tools = []
        function_declarations = []
        has_google_search = False

        for tool in tools:
            # Check if tool has google search capability
            if hasattr(tool, "name") and "google_search" in tool.name.lower():
                has_google_search = True
            elif isinstance(tool, Tool):
                function_declarations.append(self._convert_tool(tool))
            elif isinstance(tool, dict):
                google_tools.append(tool)
            else:
                raise ValueError(f"Unknown tool type: {type(tool)}")

        if function_declarations:
            google_tools.append(types.Tool(function_declarations=function_declarations))

        if has_google_search:
            google_tools.append(types.Tool(google_search=types.GoogleSearch()))

        return google_tools if google_tools else None

    def _convert_tool_choice(
        self, tool_choice: Literal["auto", "required", "none"] | list[str]
    ) -> types.ToolConfig:
        adjusted_tool_choice: types.ToolConfig
        if isinstance(tool_choice, list):
            adjusted_tool_choice = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",  # type: ignore
                    allowed_function_names=tool_choice,
                )
            )
        elif tool_choice == "required":
            adjusted_tool_choice = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")  # type: ignore
            )
        elif tool_choice == "none":
            adjusted_tool_choice = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="NONE")  # type: ignore
            )
        elif tool_choice == "auto":
            adjusted_tool_choice = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")  # type: ignore
            )
        return adjusted_tool_choice

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
        """Implementation of the abstract _invoke method"""
        if tools is None:
            tools = []
        contents = self._memory_to_contents(None, input, memory)

        tool_map = {tool.name: tool for tool in tools if isinstance(tool, Tool)}

        prepared_tools = self._prepare_tools(tools)
        config = types.GenerateContentConfig(
            temperature=temperature or self.temperature,
            system_instruction=system_prompt or self.system_prompt,
            max_output_tokens=max_tokens or None,
            tools=prepared_tools,  # type: ignore
            tool_config=self._convert_tool_choice(tool_choice)
            if tools and any(isinstance(tool, Tool) for tool in tools)
            else None,
            **kwargs,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,  # type: ignore
            config=config,  # type: ignore
        )
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
        """Implementation of the abstract _invoke method"""
        if tools is None:
            tools = []
        contents = self._memory_to_contents(None, input, memory)

        tool_map = {tool.name: tool for tool in tools if isinstance(tool, Tool)}

        prepared_tools = self._prepare_tools(tools)
        config = types.GenerateContentConfig(
            temperature=temperature or self.temperature,
            system_instruction=system_prompt or self.system_prompt,
            max_output_tokens=max_tokens or None,
            tools=prepared_tools,  # type: ignore
            tool_config=self._convert_tool_choice(tool_choice)
            if tools and any(isinstance(tool, Tool) for tool in tools)
            else None,
            **kwargs,
        )

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,  # type: ignore
            config=config,  # type: ignore
        )
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
        """Implementation of the abstract _stream_invoke method"""
        if tools is None:
            tools = []
        contents = self._memory_to_contents(None, input, memory)

        prepared_tools = self._prepare_tools(tools)
        config = types.GenerateContentConfig(
            temperature=temperature or self.temperature,
            system_instruction=system_prompt or self.system_prompt,
            max_output_tokens=max_tokens or None,
            tools=prepared_tools,  # type: ignore
            tool_config=self._convert_tool_choice(tool_choice)
            if tools and any(isinstance(tool, Tool) for tool in tools)
            else None,
            **kwargs,
        )

        message_text = ""
        thought_block = ThoughtBlock(content="")

        usage = TokenUsage()

        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,  # type: ignore
            config=config,
        ):
            usage += TokenUsage(
                prompt_tokens=chunk.usage_metadata.prompt_token_count or 0,
                completion_tokens=chunk.usage_metadata.candidates_token_count or 0,
                cached_tokens=chunk.usage_metadata.cached_content_token_count or 0,
            )
            if not chunk.candidates:
                raise ValueError("No candidates in response")

            finish_reason = chunk.candidates[0].finish_reason
            stop_reason = (
                finish_reason.value.lower()
                if finish_reason is not None
                else finish_reason
            )

            if not chunk.candidates[0].content:
                raise ValueError("No content in response")

            if not chunk.candidates[0].content.parts:
                yield ClientResponse(
                    content=[],
                    delta=chunk.text or "",
                    stop_reason=stop_reason,
                    usage=usage,
                )
                continue

            for part in chunk.candidates[0].content.parts:
                if not part.text:
                    continue
                elif hasattr(part, "thought") and part.thought:
                    thought_block.content += part.text
                else:  # If it's not a thought, it's a message
                    if part.text:
                        message_text += str(chunk.text or "")

                        yield ClientResponse(
                            content=[],
                            delta=chunk.text or "",
                            stop_reason=stop_reason,
                        )
        yield ClientResponse(
            content=[TextBlock(content=message_text)],
            stop_reason=stop_reason,
            usage=usage,
        )

    async def _a_stream_invoke(
        self,
        input: str,
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        tool_choice: Literal["auto", "required", "none"] | list[str] = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncIterator[ClientResponse]:
        """Implementation of the abstract _a_stream_invoke method for Google"""
        if tools is None:
            tools = []
        contents = self._memory_to_contents(None, input, memory)

        prepared_tools = self._prepare_tools(tools)
        config = types.GenerateContentConfig(
            temperature=temperature or self.temperature,
            system_instruction=system_prompt or self.system_prompt,
            max_output_tokens=max_tokens or None,
            tools=prepared_tools,  # type: ignore
            tool_config=self._convert_tool_choice(tool_choice)
            if tools and any(isinstance(tool, Tool) for tool in tools)
            else None,
            **kwargs,
        )

        usage = TokenUsage()
        message_text = ""
        thought_block = ThoughtBlock(content="")
        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=contents,  # type: ignore
            config=config,
        ):  # type: ignore
            usage += TokenUsage(
                prompt_tokens=chunk.usage_metadata.prompt_token_count or 0,
                completion_tokens=chunk.usage_metadata.candidates_token_count or 0,
                cached_tokens=chunk.usage_metadata.cached_content_token_count or 0,
            )

            finish_reason = chunk.candidates[0].finish_reason
            stop_reason = (
                finish_reason.value.lower()
                if finish_reason is not None
                else finish_reason
            )

            # Handle the case where the response has no parts
            if not chunk.candidates[0].content.parts:
                yield ClientResponse(
                    content=[],
                    delta=chunk.text or "",
                    stop_reason=stop_reason,
                )
                continue

            for part in chunk.candidates[0].content.parts:
                if not part.text:
                    continue
                elif hasattr(part, "thought") and part.thought:
                    thought_block.content += part.text
                else:  # If it's not a thought, it's a message
                    if part.text:
                        message_text += chunk.text or ""
                        yield ClientResponse(
                            content=[],
                            delta=chunk.text or "",
                            stop_reason=stop_reason,
                        )
        yield ClientResponse(
            content=[TextBlock(content=message_text)],
            stop_reason=stop_reason,
            usage=usage,
        )

    def _structured_response(
        self,
        input: str,
        output_cls: type[Model],
        memory: Memory | None,
        temperature: float | None,
        max_tokens: int,
        system_prompt: str | None,
        tools: list[Tool] | None,
        tool_choice: Literal["auto", "required", "none"] | list[str] = "auto",
        **kwargs,
    ) -> ClientResponse:
        """Implementation of the abstract _structured_response method"""
        contents = self._memory_to_contents(self.system_prompt, input, memory)

        prepared_tools = self._prepare_tools(tools)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,  # type: ignore
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                tools=prepared_tools,  # type: ignore
                tool_config=self._convert_tool_choice(tool_choice)
                if tools and any(isinstance(tool, Tool) for tool in tools)
                else None,
                response_schema=(
                    output_cls.model_json_schema()
                    if hasattr(output_cls, "model_json_schema")
                    else output_cls
                ),
            ),
        )
        if not response or not response.candidates:
            raise ValueError("No response from Google GenAI")

        structured_data = output_cls.model_validate_json(str(response.text))
        return ClientResponse(
            content=[StructuredBlock(content=structured_data)],
            stop_reason=response.candidates[0].finish_reason.value.lower()
            if response.candidates[0].finish_reason
            else None,
            usage=TokenUsage(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
                cached_tokens=response.usage_metadata.cached_content_token_count or 0,
            ),
        )

    async def _a_structured_response(
        self,
        input: str,
        output_cls: type[Model],
        memory: Memory | None,
        temperature: float | None,
        max_tokens: int,
        system_prompt: str | None,
        tools: list[Tool] | None,
        tool_choice: Literal["auto", "required", "none"] | list[str] = "auto",
        **kwargs,
    ) -> ClientResponse:
        """Implementation of the abstract _structured_response method"""
        contents = self._memory_to_contents(self.system_prompt, input, memory)
        prepared_tools = self._prepare_tools(tools)
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,  # type: ignore
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                tools=prepared_tools,  # type: ignore
                tool_config=self._convert_tool_choice(tool_choice)
                if tools and any(isinstance(tool, Tool) for tool in tools)
                else None,
                response_schema=(
                    output_cls.model_json_schema()
                    if hasattr(output_cls, "model_json_schema")
                    else output_cls
                ),
            ),
        )

        if not response or not response.candidates:
            raise ValueError("No response from Google GenAI")

        structured_data = output_cls.model_validate_json(str(response.text))
        return ClientResponse(
            content=[StructuredBlock(content=structured_data)],
            stop_reason=response.candidates[0].finish_reason.value.lower()
            if response.candidates[0].finish_reason
            else None,
            usage=TokenUsage(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
                cached_tokens=response.usage_metadata.cached_content_token_count or 0,
            ),
        )

    def _embed(
        self,
        text: str | list[str],
        model_name: str | None,
        task_type: str = "RETRIEVAL_DOCUMENT",
        output_dimensionality: int = 768,
        title: str | None = None,
        **kwargs,
    ) -> list[float] | list[list[float] | None]:
        """Embed a text using the model"""
        response = self.client.models.embed_content(
            model=model_name or self.model_name,
            contents=text,  # type: ignore
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=output_dimensionality,
                title=title,
                **kwargs,
            ),
        )
        # Extract the embedding values from the response
        if not response.embeddings:
            return []

        embeddings = [embedding.values for embedding in response.embeddings]

        if isinstance(text, str) and embeddings[0]:
            return embeddings[0]

        return embeddings

    async def _a_embed(
        self,
        text: str | list[str],
        model_name: str | None,
        task_type: str = "RETRIEVAL_DOCUMENT",
        output_dimensionality: int = 768,
        title: str | None = None,
        **kwargs,
    ) -> list[float] | list[list[float] | None]:
        """Embed a text using the model"""
        response = await self.client.aio.models.embed_content(
            model=model_name or self.model_name,
            contents=text,  # type: ignore
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=output_dimensionality,
                title=title,
                **kwargs,
            ),
        )
        # Extract the embedding values from the response
        if not response.embeddings:
            return []
        embeddings = [embedding.values for embedding in response.embeddings]

        if isinstance(text, str) and embeddings[0]:
            return embeddings[0]

        return embeddings

    def _response_to_client_response(
        self, response, tool_map: dict[str, Tool] | None = None
    ) -> ClientResponse:
        blocks = []
        # Handle function calls if present
        if hasattr(response, "function_calls") and response.function_calls:
            for fc in response.function_calls:
                if not tool_map:
                    raise ValueError("Tool map is required")

                tool = tool_map.get(fc.name, None)
                if not tool:
                    raise ValueError(f"Tool {fc.name} not found in tool map")

                blocks.append(
                    FunctionCallBlock(
                        name=fc.name,
                        arguments=fc.args,
                        id=f"fc_{id(fc)}",
                        tool=tool,
                    )
                )
        else:
            if hasattr(response, "text") and response.text:
                blocks.append(TextBlock(content=response.text))

        if hasattr(response, "candidates") and response.candidates:
            for part in response.candidates[0].content.parts:
                if not part.text:
                    continue
                if hasattr(part, "thought") and part.thought:
                    blocks.append(ThoughtBlock(content=part.text))

        usage_metadata = getattr(response, "usage_metadata", None)
        return ClientResponse(
            content=blocks,
            stop_reason=(response.candidates[0].finish_reason.value.lower())
            if hasattr(response, "candidates") and response.candidates
            else None,
            usage=TokenUsage(
                prompt_tokens=usage_metadata.prompt_token_count or 0,
                completion_tokens=usage_metadata.candidates_token_count or 0,
                cached_tokens=usage_metadata.cached_content_token_count or 0,
            ),
        )
