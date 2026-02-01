import base64

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

from google.genai import types


class GoogleMemoryAdapter(MemoryAdapter):
    def _turn_to_message(self, turn: Turn) -> dict:
        content = []
        for block in turn:
            block_dict = {}

            match block:
                case TextBlock():
                    block_dict = {"text": block.content}
                case FunctionCallBlock():
                    block_dict = {
                        "function_call": {"name": block.name, "args": block.arguments}
                    }
                case FunctionCallResultBlock():
                    block_dict = types.Part.from_function_response(
                        name=block.tool.name,
                        response={"result": block.result},
                    )
                case StructuredBlock():
                    block_dict = {"text": str(block.content)}
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

            content.append(block_dict)

        return {
            "role": turn.role.google_role,
            "parts": (content),
        }

    def _process_audio_block(self, block: MediaBlock) -> types.Part:
        match block.media.source_type:
            case "raw":
                return types.Part.from_bytes(
                    data=block.media.source,
                    mime_type="audio/mp3",
                )

            case "path":
                with open(block.media.source, "rb") as f:
                    audio_bytes = f.read()

                return types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/mp3",
                )

            case _:
                raise NotImplementedError(
                    f"Unsupported media source type: {block.media.source_type} for audio, source type supported: raw, path"
                )

    def _process_pdf_block(self, block: MediaBlock) -> types.Part | dict:
        match block.media.source_type:
            case "raw":
                return types.Part.from_bytes(
                    data=block.media.source,
                    mime_type="application/pdf",
                )
            case "base64":
                return {
                    "inline_data": {
                        "mime_type": "application/pdf",
                        "data": block.media.source,
                    }
                }
            case "path":
                with open(block.media.source, "rb") as f:
                    pdf_bytes = f.read()

                return {
                    "inline_data": {
                        "mime_type": "application/pdf",
                        "data": pdf_bytes,
                    }
                }
            case "url":
                return types.Part.from_uri(
                    file_uri=block.media.source,
                    mime_type="application/pdf",
                )

            case _:
                raise NotImplementedError(
                    f"Unsupported media source type: {block.media.source_type} only supported: raw, base64, path"
                )

    def _process_image_block(self, block: MediaBlock) -> dict:
        match block.media.source_type:
            case "url":
                return types.Part.from_uri(
                    file_uri=block.media.source,
                    mime_type=f"image/{block.media.extension}",
                )  # type: ignore
            case "base64":
                return {
                    "inline_data": {
                        "mime_type": f"image/{block.media.extension}",
                        "data": block.media.source,
                    }
                }
            case "path":
                with open(block.media.source, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                return {
                    "inline_data": {
                        "mime_type": f"image/{block.media.extension}",
                        "data": base64_image,
                    }
                }
            case _:
                raise NotImplementedError(
                    f"Unsupported media source type: {block.media.source_type} for image, only url, base64, path are supported"
                )

    def _text_to_message(self, text: str, role: ROLE) -> dict:
        return {"role": role.google_role, "parts": [{"text": text}]}
