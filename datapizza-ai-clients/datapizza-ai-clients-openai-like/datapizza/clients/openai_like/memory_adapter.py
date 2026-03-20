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


class OpenAILikeMemoryAdapter(MemoryAdapter):
    def _turn_to_message(self, turn: Turn) -> dict:
        content_blocks: list[dict] = []
        tool_calls = []
        tool_call_id = None
        text_parts: list[str] = []

        for block in turn:
            block_dict = {}

            match block:
                case TextBlock():
                    text_parts.append(block.content)
                case StructuredBlock():
                    text_parts.append(str(block.content))
                case FunctionCallBlock():
                    tool_calls.append(
                        {
                            "id": block.id,
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.arguments),
                            },
                            "type": "function",
                        }
                    )
                case FunctionCallResultBlock():
                    tool_call_id = block.id
                    text_parts.append(block.result)
                case MediaBlock():
                    match block.media.media_type:
                        case "image":
                            block_dict = self._process_image_block(block)
                        case "pdf":
                            block_dict = self._process_pdf_block(block)
                        case "audio":
                            block_dict = self._process_audio_block(block)
                        case _:
                            raise NotImplementedError(
                                f"Unsupported media type: {block.media.media_type}"
                            )

            if block_dict:
                content_blocks.append(block_dict)

        text_content = "".join(text_parts) if text_parts else None

        if content_blocks:
            if text_content:
                content_blocks.insert(0, {"type": "text", "text": text_content})
            content_value = content_blocks
        else:
            content_value = text_content if text_content is not None else ""

        message = {"role": turn.role.value, "content": content_value}

        if tool_calls:
            message["tool_calls"] = tool_calls
        if tool_call_id:
            message["tool_call_id"] = tool_call_id

        return message

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
                    f"Unsupported media source type: {block.media.source_type} for audio, source type supported: raw, path"
                )

    def _text_to_message(self, text: str, role: ROLE) -> dict:
        return {"role": role.value, "content": text}

    def _process_pdf_block(self, block: MediaBlock) -> dict:
        match block.media.source_type:
            case "base64":
                return {
                    "type": "file",
                    "file": {
                        "filename": "file.pdf",
                        "file_data": f"data:application/{block.media.extension};base64,{block.media.source}",
                    },
                }
            case "path":
                with open(block.media.source, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode("utf-8")
                return {
                    "type": "file",
                    "file": {
                        "filename": "file.pdf",
                        "file_data": f"data:application/{block.media.extension};base64,{base64_pdf}",
                    },
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
