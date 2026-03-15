import base64
import json
from typing import Any

from datapizza.memory.memory import Turn
from datapizza.memory.memory_adapter import MemoryAdapter
from datapizza.type import (
    ROLE,
    FunctionCallBlock,
    FunctionCallResultBlock,
    MediaBlock,
    StructuredBlock,
    TextBlock,
)


class AnthropicMemoryAdapter(MemoryAdapter):
    """Adapter for converting Memory objects to Anthropic API message format"""

    @staticmethod
    def _normalize_tool_result_content(result: Any) -> str | list[dict[str, Any]]:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return json.dumps(result)
        return str(result)

    def _turn_to_message(self, turn: Turn) -> dict:
        content = []
        for block in turn:
            block_dict = {}

            match block:
                case TextBlock():
                    block_dict = {"type": "text", "text": block.content}
                case FunctionCallBlock():
                    block_dict = {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.arguments,
                    }

                case FunctionCallResultBlock():
                    block_dict = {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": self._normalize_tool_result_content(block.result),
                    }
                case StructuredBlock():
                    block_dict = {
                        "type": "text",
                        "text": str(block.content),
                    }
                case MediaBlock():
                    match block.media.media_type:
                        case "image":
                            block_dict = self._process_image_block(block)
                        case "pdf":
                            block_dict = self._process_pdf_block(block)

                        case _:
                            raise NotImplementedError(
                                f"Unsupported media type: {block.media.media_type}"
                            )

            content.append(block_dict)

        if all(isinstance(block, dict) for block in content) and all(
            list(block.keys()) == ["type", "text"] for block in content
        ):
            content = "".join([block["text"] for block in content])

        return {
            "role": turn.role.anthropic_role,
            "content": (content),
        }

    def _text_to_message(self, text: str, role: ROLE) -> dict:
        """Convert text and role to Anthropic message format"""
        # Anthropic uses 'user', 'assistant', and 'system' roles

        return {"role": role.anthropic_role, "content": text}

    def _process_pdf_block(self, block: MediaBlock) -> dict:
        match block.media.source_type:
            case "url":
                return {
                    "type": "document",
                    "source": {
                        "type": "url",
                        "url": block.media.source,
                    },
                }

            case "base64":
                return {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": block.media.source,
                    },
                }

            case "path":
                with open(block.media.source, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode("utf-8")
                return {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64_pdf,
                    },
                }

            case _:
                raise NotImplementedError("Source type not supported")

    def _process_image_block(self, block: MediaBlock) -> dict:
        match block.media.source_type:
            case "url":
                return {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": block.media.source,
                    },
                }

            case "base64":
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{block.media.extension}",
                        "data": block.media.source,
                    },
                }

            case "path":
                with open(block.media.source, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{block.media.extension}",
                        "data": base64_image,
                    },
                }
            case _:
                raise NotImplementedError(
                    f"Unsupported media type: {block.media.media_type}"
                )
