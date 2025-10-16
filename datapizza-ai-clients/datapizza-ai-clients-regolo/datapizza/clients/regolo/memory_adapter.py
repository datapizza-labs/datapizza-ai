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


class RegoloMemoryAdapter(MemoryAdapter):
    """Memory adapter for Regolo.ai using OpenAI-compatible format."""

    def _turn_to_message(self, turn: Turn) -> dict:
        content = []
        tool_calls = []
        tool_call_id = None

        for block in turn:
            block_dict = {}

            match block:
                case TextBlock():
                    # Only add text block if content is not None/empty
                    if block.content:
                        block_dict = {"type": "text", "text": block.content}
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
                    # Only add result if it's not None/empty
                    if block.result:
                        block_dict = {"type": "text", "text": block.result}
                case StructuredBlock():
                    content_str = str(block.content)
                    if content_str:
                        block_dict = {"type": "text", "text": content_str}
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
                                f"Unsupported media type: "
                                f"{block.media.media_type}"
                            )

            if block_dict:
                content.append(block_dict)

        # Build the message structure
        # If this turn contains a tool result, use "tool" role
        role = "tool" if tool_call_id else turn.role.value
        
        messages = {
            "role": role,
        }
        
        # For tool role, content must be a string, not an array
        if tool_call_id:
            messages["tool_call_id"] = tool_call_id
            # Tool messages require string content
            messages["content"] = content[0]["text"] if content else ""
        else:
            # Content can be string or array depending on complexity
            # If there's only text content, use array format (supports multimodal)
            # If no content but there are tool_calls, omit content for assistant
            if content:
                messages["content"] = content  # type: ignore
            elif not tool_calls:
                # No content and no tool_calls - use empty string
                messages["content"] = ""
            
            if tool_calls:
                messages["tool_calls"] = tool_calls  # type: ignore

        return messages

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
                base64_audio = base64.b64encode(
                    block.media.source
                ).decode("utf-8")
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64_audio,
                        "format": block.media.extension,
                    },
                }

            case _:
                raise NotImplementedError(
                    f"Unsupported media source type: "
                    f"{block.media.source_type} for audio, "
                    f"source type supported: raw, path"
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
                        "file_data": (
                            f"data:application/{block.media.extension};"
                            f"base64,{block.media.source}"
                        ),
                    },
                }
            case "path":
                with open(block.media.source, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode("utf-8")
                return {
                    "type": "file",
                    "file": {
                        "filename": "file.pdf",
                        "file_data": (
                            f"data:application/{block.media.extension};"
                            f"base64,{base64_pdf}"
                        ),
                    },
                }

            case _:
                raise NotImplementedError(
                    f"Unsupported media source type: "
                    f"{block.media.source_type}"
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
                        "url": (
                            f"data:image/{block.media.extension};"
                            f"base64,{block.media.source}"
                        )
                    },
                }

            case "path":
                with open(block.media.source, "rb") as image_file:
                    base64_image = base64.b64encode(
                        image_file.read()
                    ).decode("utf-8")
                    return {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:image/{block.media.extension};"
                                f"base64,{base64_image}"
                            )
                        },
                    }

            case _:
                raise ValueError(
                    f"Unsupported media source type: "
                    f"{block.media.source_type}"
                )

