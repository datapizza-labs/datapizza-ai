import warnings

from pydantic import BaseModel, Field

from datapizza.type import (
    Block,
    FunctionCallBlock,
    StructuredBlock,
    TextBlock,
    ThoughtBlock,
)


class TokenUsage(BaseModel):
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    cached_tokens: int = Field(default=0)
    thinking_tokens: int = Field(default=0)

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            thinking_tokens=self.thinking_tokens + other.thinking_tokens,
        )


class ClientResponse:
    """
    A class for storing the response from a client.
    Contains a list of blocks that can be text, function calls, or structured data,
    maintaining the order in which they were generated.

    Args:
        content (List[Block]): A list of blocks.
        delta (str, optional): The delta of the response. Used for streaming responses.
        usage (TokenUsage, optional): Aggregated token usage.
        stop_reason (str, optional): Stop reason.

    """

    def __init__(
        self,
        *,
        content: list[Block],
        delta: str | None = None,
        stop_reason: str | None = None,
        usage: TokenUsage | None = None,
        # Deprecated per-field args for backward compatibility:
        prompt_tokens_used: int | None = None,
        completion_tokens_used: int | None = None,
        cached_tokens_used: int | None = None,
        thinking_tokens_used: int | None = None,
    ):
        self.content = content
        self.delta = delta
        self.stop_reason = stop_reason

        if any(
            v is not None
            for v in (
                prompt_tokens_used,
                completion_tokens_used,
                cached_tokens_used,
                thinking_tokens_used,
            )
        ):
            warnings.warn(
                "ClientResponse: per-field token args are deprecated and ignored when `usage` is provided. "
                "Pass a TokenUsage via `usage`.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.usage = usage or TokenUsage(
            prompt_tokens=prompt_tokens_used or 0,
            completion_tokens=completion_tokens_used or 0,
            cached_tokens=cached_tokens_used or 0,
            thinking_tokens=thinking_tokens_used or 0,
        )

    def __eq__(self, other):
        return isinstance(other, ClientResponse) and self.content == other.content

    @property
    def prompt_tokens_used(self) -> int:
        return self.usage.prompt_tokens

    @property
    def completion_tokens_used(self) -> int:
        return self.usage.completion_tokens

    @property
    def cached_tokens_used(self) -> int:
        return self.usage.cached_tokens

    @property
    def thinking_tokens_used(self) -> int:
        return self.usage.thinking_tokens

    @property
    def text(self) -> str:
        """Returns concatenated text from all TextBlocks in order"""
        return "\n".join(
            block.content for block in self.content if isinstance(block, TextBlock)
        )

    @property
    def thoughts(self) -> str:
        """Returns all thoughts in order"""
        return "\n".join(
            block.content for block in self.content if isinstance(block, ThoughtBlock)
        )

    @property
    def first_text(self) -> str | None:
        """Returns the content of the first TextBlock or None"""
        text_block = next(
            (item for item in self.content if isinstance(item, TextBlock)), None
        )
        return text_block.content if text_block else None

    @property
    def function_calls(self) -> list[FunctionCallBlock]:
        """Returns all function calls in order"""
        return [item for item in self.content if isinstance(item, FunctionCallBlock)]

    @property
    def structured_data(self) -> list[BaseModel]:
        """Returns all structured data in order"""
        return [
            item.content for item in self.content if isinstance(item, StructuredBlock)
        ]

    def is_pure_text(self) -> bool:
        """Returns True if response contains only TextBlocks"""
        return all(isinstance(block, TextBlock) for block in self.content)

    def is_pure_function_call(self) -> bool:
        """Returns True if response contains only FunctionCallBlocks"""
        return all(isinstance(block, FunctionCallBlock) for block in self.content)

    def __str__(self) -> str:
        return f"ClientResponse(content={self.content}, delta={self.delta}, stop_reason={self.stop_reason})"
