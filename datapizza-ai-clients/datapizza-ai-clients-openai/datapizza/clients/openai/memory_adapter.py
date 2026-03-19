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

from openai.types.responses import ResponseFunctionToolCall


class OpenAIMemoryAdapter(MemoryAdapter):
    def _turn_to_message(self, turn: Turn) -> dict:
        content = []
        tool_calls = []
        tool_call_id = None
        response_function_tool_call = None

        for block in turn:
            block_dict = {}

            match block:
                case TextBlock():
                    block_dict = {
                        "type": "input_text"
                        if turn.role == ROLE.USER
                        else "output_text",
                        "text": block.content,
                    }
                case FunctionCallBlock():
                    return ResponseFunctionToolCall(
                        call_id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.arguments),
                        type="function_call",
                        # id="fc_" + block.id,
                        status="completed",
                    )
                    # block_dict = {
                    #    "id": block.id,
                    #    "name": block.name,
                    #    "arguments": json.dumps(block.arguments),
                    #    "type": "function_call",
                    # }
                case FunctionCallResultBlock():
                    tool_call_id = block.id
                    return {
                        "type": "function_call_output",
                        "call_id": block.id,
                        "output": block.result,
                    }

                case StructuredBlock():
                    block_dict = {
                        "type": "input_text"
                        if turn.role == ROLE.USER
                        else "output_text",
                        "text": str(block.content),
                    }
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
                content.append(block_dict)

        messages = {
            "role": turn.role.value,
            "content": (content),
        }

        if tool_calls:
            messages["tool_calls"] = tool_calls

        if tool_call_id:
            messages["tool_call_id"] = tool_call_id

        if response_function_tool_call:
            return response_function_tool_call

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
                    "type": "input_image",
                    "image_url": block.media.source,
                }

            case "base64":
                return {
                    "type": "input_image",
                    "image_url": f"data:image/{block.media.extension};base64,{block.media.source}",
                }

            case "path":
                with open(block.media.source, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                    return {
                        "type": "input_image",
                        "image_url": f"data:image/{block.media.extension};base64,{base64_image}",
                    }

            case _:
                raise ValueError(
                    f"Unsupported media source type: {block.media.source_type}"
                )
