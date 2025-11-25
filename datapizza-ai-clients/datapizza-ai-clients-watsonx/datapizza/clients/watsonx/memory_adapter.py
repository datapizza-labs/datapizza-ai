import base64
import json
import logging
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

log = logging.getLogger(__name__)


class WatsonXMemoryAdapter(MemoryAdapter):
    def _turn_to_message(self, turn: Turn) -> dict:
        content: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []

        for block in turn:
            match block:
                case TextBlock():
                    content.append({"type": "text", "text": block.content})

                case FunctionCallBlock():
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.arguments),
                            },
                        }
                    )

                case FunctionCallResultBlock():
                    # Tool output is a separate message type usually
                    return {
                        "role": "tool",
                        "tool_call_id": block.id,
                        "content": str(block.result),
                    }

                case StructuredBlock():
                    content.append({"type": "text", "text": str(block.content)})

                case MediaBlock():
                    match block.media.media_type:
                        case "image":
                            content.append(self._process_image_block(block))
                        case "pdf":
                            content.append(self._process_pdf_block(block))
                        case "audio":
                            content.append(self._process_audio_block(block))
                        case _:
                            log.warning(
                                f"Unsupported media type: {block.media.media_type}"
                            )

        message: dict[str, Any] = {
            "role": turn.role.value,
        }

        if tool_calls:
            message["tool_calls"] = tool_calls
            if content:
                message["content"] = content
            else:
                message["content"] = ""
        else:
            # Optimization: if single text block, use string content
            if len(content) == 1 and content[0]["type"] == "text":
                message["content"] = content[0]["text"]
            else:
                message["content"] = content

        return message

    def _text_to_message(self, text: str, role: ROLE) -> dict:
        return {"role": role.value, "content": text}

    def _process_audio_block(self, block: MediaBlock) -> dict:
        match block.media.source_type:
            case "path":
                with open(block.media.source, "rb") as f:
                    base64_audio = base64.b64encode(f.read()).decode("utf-8")
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64_audio,
                        "format": block.media.extension,
                    },
                }

            case "base_64":
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": block.media.source,
                        "format": block.media.extension,
                    },
                }
            case "raw":
                base64_audio = base64.b64encode(block.media.source).decode("utf-8")
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64_audio,
                        "format": block.media.extension,
                    },
                }

            case _:
                raise NotImplementedError(
                    f"Unsupported media source type: {block.media.source_type} for audio"
                )

    def _process_pdf_block(self, block: MediaBlock) -> dict:
        match block.media.source_type:
            case "base64":
                return {
                    "type": "input_file",
                    "filename": "file.pdf",
                    "file_data": f"data:application/{block.media.extension};base64,{block.media.source}",
                }
            case "path":
                with open(block.media.source, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode("utf-8")
                return {
                    "type": "input_file",
                    "filename": "file.pdf",
                    "file_data": f"data:application/{block.media.extension};base64,{base64_pdf}",
                }

            case _:
                raise NotImplementedError(
                    f"Unsupported media source type: {block.media.source_type}"
                )

    def _process_image_block(self, block: MediaBlock) -> dict:
        match block.media.source_type:
            case "url":
                return {
                    "type": "image_url",
                    "image_url": {"url": block.media.source},
                }

            case "base64":
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{block.media.extension};base64,{block.media.source}"
                    },
                }

            case "path":
                with open(block.media.source, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                    return {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{block.media.extension};base64,{base64_image}"
                        },
                    }

            case _:
                raise ValueError(
                    f"Unsupported media source type: {block.media.source_type}"
                )
