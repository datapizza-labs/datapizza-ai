import base64
import json

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


class BedrockMemoryAdapter(MemoryAdapter):
    """Adapter for converting Memory objects to AWS Bedrock Converse API message format

    The Bedrock Converse API uses a message format similar to Anthropic's Claude API.
    """

    def _turn_to_message(self, turn: Turn) -> dict:
        """Convert a Turn to Bedrock message format"""
        content = []

        for block in turn:
            block_dict = {}

            match block:
                case TextBlock():
                    block_dict = {"text": block.content}

                case FunctionCallBlock():
                    # Bedrock uses toolUse format
                    block_dict = {
                        "toolUse": {
                            "toolUseId": block.id,
                            "name": block.name,
                            "input": block.arguments,
                        }
                    }

                case FunctionCallResultBlock():
                    # Bedrock uses toolResult format
                    block_dict = {
                        "toolResult": {
                            "toolUseId": block.id,
                            "content": [{"text": str(block.result)}],
                        }
                    }

                case StructuredBlock():
                    # Convert structured content to text
                    block_dict = {
                        "text": json.dumps(block.content)
                        if not isinstance(block.content, str)
                        else block.content,
                    }

                case MediaBlock():
                    match block.media.media_type:
                        case "image":
                            block_dict = self._process_image_block(block)
                        case "pdf":
                            block_dict = self._process_document_block(block)
                        case _:
                            raise NotImplementedError(
                                f"Unsupported media type: {block.media.media_type}"
                            )

            content.append(block_dict)

        # Bedrock expects content as a list
        return {
            "role": self._convert_role(turn.role),
            "content": content,
        }

    def _text_to_message(self, text: str, role: ROLE) -> dict:
        """Convert text and role to Bedrock message format"""
        return {
            "role": self._convert_role(role),
            "content": [{"text": text}],
        }

    def _convert_role(self, role: ROLE) -> str:
        """Convert ROLE to Bedrock role string

        Bedrock Converse API supports 'user' and 'assistant' roles
        """
        if role.value == "user":
            return "user"
        elif role.value == "assistant":
            return "assistant"
        else:
            # Default to user for system or other roles
            return "user"

    def _process_document_block(self, block: MediaBlock) -> dict:
        """Process document (PDF) blocks for Bedrock"""
        match block.media.source_type:
            case "base64":
                return {
                    "document": {
                        "format": "pdf",
                        "name": "document.pdf",
                        "source": {"bytes": base64.b64decode(block.media.source)},
                    }
                }

            case "path":
                with open(block.media.source, "rb") as f:
                    pdf_bytes = f.read()
                return {
                    "document": {
                        "format": "pdf",
                        "name": block.media.source.split("/")[-1],
                        "source": {"bytes": pdf_bytes},
                    }
                }

            case "url":
                raise NotImplementedError(
                    "Bedrock Converse API does not support document URLs directly. "
                    "Please download and provide as bytes."
                )

            case _:
                raise NotImplementedError(
                    f"Unsupported source type: {block.media.source_type}"
                )

    def _process_image_block(self, block: MediaBlock) -> dict:
        """Process image blocks for Bedrock"""
        match block.media.source_type:
            case "base64":
                return {
                    "image": {
                        "format": block.media.extension or "png",
                        "source": {"bytes": base64.b64decode(block.media.source)},
                    }
                }

            case "path":
                with open(block.media.source, "rb") as image_file:
                    image_bytes = image_file.read()
                return {
                    "image": {
                        "format": block.media.extension or "png",
                        "source": {"bytes": image_bytes},
                    }
                }

            case "url":
                raise NotImplementedError(
                    "Bedrock Converse API does not support image URLs directly. "
                    "Please download and provide as bytes."
                )

            case _:
                raise NotImplementedError(
                    f"Unsupported source type: {block.media.source_type}"
                )
