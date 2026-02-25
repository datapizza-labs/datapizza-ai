import hashlib
import logging
import uuid
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from datapizza.tools.tools import Tool

log = logging.getLogger(__name__)

Model = TypeVar("Model", bound=BaseModel)


class ROLE(Enum):
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    TOOL = "tool"

    @property
    def google_role(self) -> str:
        if self == ROLE.USER:
            return "user"
        elif self == ROLE.ASSISTANT or self == ROLE.SYSTEM:
            return "model"
        elif self == ROLE.TOOL:
            return "tool"
        else:
            raise ValueError(f"Unknown role: {self}")

    @property
    def anthropic_role(self) -> str:
        if self == ROLE.USER:
            return "user"
        elif self == ROLE.ASSISTANT:
            return "assistant"
        elif self == ROLE.SYSTEM:
            return "model"
        elif self == ROLE.TOOL:
            return "assistant"
        else:
            raise ValueError(f"Unknown role: {self}")


class Block:
    """
    A class for storing the response from a client.
    """

    def __init__(self, type: str):
        self.type = type

    @classmethod
    def from_dict(cls, data: dict):
        match data["type"]:
            case "text":
                return TextBlock(content=data.get("content", ""))
            case "thought":
                return ThoughtBlock(content=data.get("content", ""))
            case "function":
                tool = Tool.tool_from_dict(data.get("tool"))
                return FunctionCallBlock(
                    id=data.get("id", ""),
                    arguments=data.get("arguments", {}),
                    name=data.get("name", ""),
                    tool=tool,
                    thought_signature=data.get("thought_signature"),
                )
            case "function_call_result":
                tool = Tool.tool_from_dict(data.get("tool"))
                return FunctionCallResultBlock(
                    id=data.get("id", ""), tool=tool, result=data.get("result", "")
                )
            case "structured":
                logging.warning(
                    "Structured Blocks clouldn't load BaseModel, dict loaded instead"
                )
                return StructuredBlock(**data)
            case "media":
                return MediaBlock.from_dict(data)
            case _:
                raise ValueError(f"Invalid block type: {data['type']}")

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert the block to a dictionary for JSON serialization."""
        pass


class TextBlock(Block):
    """
    A class for storing the text response from a client.
    """

    def __init__(self, content: str, type: str = "text"):
        """
        Initialize a TextBlock object.

        Args:
            content (str): The content of the text block.
            type (str, optional): The type of the text block. Defaults to "text".
        """
        self.content = content
        super().__init__(type)

    def __eq__(self, other):
        return isinstance(other, TextBlock) and self.content == other.content

    def __str__(self) -> str:
        return f"TextBlock(content={self.content})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return int(hashlib.sha256(self.content.encode("utf-8")).hexdigest(), 16)

    def to_dict(self) -> dict:
        return {"type": self.type, "content": self.content}


class ThoughtBlock(Block):
    """
    A class for storing the thought from a client.
    """

    def __init__(self, content: str, type: str = "thought"):
        """
        Initialize a ThoughtBlock object.

        Args:
            content (str): The content of the thought block.
            type (str, optional): The type of the thought block. Defaults to "thought".
        """
        self.content = content
        super().__init__(type)

    def __eq__(self, other):
        return isinstance(other, ThoughtBlock) and self.content == other.content

    def __str__(self) -> str:
        return f"ThoughtBlock(content={self.content})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return int(hashlib.sha256(self.content.encode("utf-8")).hexdigest(), 16)

    def to_dict(self) -> dict:
        return {"type": self.type, "content": self.content}


class FunctionCallBlock(Block):
    """
    A class for storing the function call from a client.
    """

    def __init__(
        self,
        id: str,
        arguments: dict[str, Any],
        name: str,
        tool: Tool,
        type: str = "function",
        thought_signature: str | None = None,
    ):
        """
        Initialize a FunctionCallBlock object.

        Args:
            id (str): The id of the function call block.
            arguments (dict[str, Any]): The arguments of the function call block.
            name (str): The name of the function call block.
            tool (Tool): The tool of the function call block.
            thought_signature (str, optional): The thought signature for Gemini 2.0+ models.
                Required for multi-turn function calling with thinking models.
        """
        self.id = id
        self.arguments = arguments
        self.name = name
        self.tool = tool
        self.thought_signature = thought_signature
        super().__init__(type)

    def __eq__(self, other):
        return (
            isinstance(other, FunctionCallBlock)
            and self.id == other.id
            and self.arguments == other.arguments
            and self.name == other.name
        )

    def __str__(self) -> str:
        return f"FunctionCallBlock(id={self.id}, arguments={self.arguments}, name={self.name}, tool={self.tool})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return int(hashlib.sha256(self.id.encode("utf-8")).hexdigest(), 16)

    def to_dict(self) -> dict:
        result = {
            "type": self.type,
            "id": self.id,
            "arguments": self.arguments,
            "name": self.name,
            "tool": self.tool.to_dict(),
        }
        if self.thought_signature:
            result["thought_signature"] = self.thought_signature
        return result


class FunctionCallResultBlock(Block):
    """
    A class for storing the function call response from a client.
    """

    def __init__(
        self,
        id: str,
        tool: Tool,
        result: str,
        type: str = "function_call_result",
    ):
        """
        Initialize a FunctionCallResultBlock object.

        Args:
            id (str): The id of the function call result block.
            tool (Tool): The tool of the function call result block.
            result (str): The result of the function call result block.
        """
        self.id = id
        self.tool = tool
        self.result = result
        super().__init__(type)

    def __hash__(self) -> int:
        return int(hashlib.sha256(self.id.encode("utf-8")).hexdigest(), 16)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "id": self.id,
            "tool": self.tool.to_dict(),
            "result": self.result,
        }


class StructuredBlock(Block):
    """
    A class for storing the structured response from a client.
    """

    def __init__(self, content: BaseModel, type: str = "structured"):
        """
        Initialize a StructuredBlock object.

        Args:
            content (BaseModel): The content of the structured block.
            type (str, optional): The type of the structured block. Defaults to "structured".
        """
        self.content = content
        super().__init__(type)

    def __hash__(self) -> int:
        return int(
            hashlib.sha256(self.content.model_dump_json().encode("utf-8")).hexdigest(),
            16,
        )

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "content": self.content.model_dump_json()
            if isinstance(self.content, BaseModel)
            else self.content,
        }


class Media:
    """
    A class for storing the media response from a client.
    """

    def __init__(
        self,
        *,
        extension: str | None = None,
        media_type: Literal["image", "video", "audio", "pdf"],
        source_type: Literal["url", "base64", "path", "pil", "raw"],
        source: Any,
        detail: str = "high",
    ):
        """
        A class for storing the media response from a client.

        arguments:
            extension (str, optional): The file extension of the media. Defaults to None.
            media_type (Literal["image", "video", "audio", "pdf"]): The type of media. Defaults to "image".
            source_type (Literal["url", "base64", "path", "pil", "raw"]): The source type of the media. Defaults to "url".
            source (Any): The source of the media. Defaults to None.
        """
        self.extension = extension
        self.media_type = media_type
        self.source_type = source_type
        self.source = source
        self.detail = detail

    def to_dict(self) -> dict:
        """Convert the media to a dictionary for JSON serialization."""
        return {
            "extension": self.extension,
            "media_type": self.media_type,
            "source_type": self.source_type,
            "source": str(self.source),  # Convert to string for JSON serialization
            "detail": self.detail,
        }


class MediaBlock(Block):
    """
    A class for storing the media response from a client.
    """

    def __init__(self, media: Media, type: str = "media"):
        """
        Initialize a MediaBlock object.

        Args:
            media (Media): The media of the media block.
            type (str, optional): The type of the media block. Defaults to "media".
        """
        self.media = media
        super().__init__(type)

    def __hash__(self) -> int:
        return int(hashlib.sha256(self.media.source.encode("utf-8")).hexdigest(), 16)

    def to_dict(self) -> dict:
        return {"type": self.type, "media": self.media.to_dict()}

    @classmethod
    def from_dict(cls, json_data):
        media_data = json_data.get("media")
        media = Media(**media_data)
        return MediaBlock(media=media)


class NodeType(Enum):
    SECTION = "section"
    PARAGRAPH = "paragraph"
    DOCUMENT = "document"
    SENTENCE = "sentence"
    PAGE = "page"
    TABLE = "table"
    FIGURE = "figure"


class Node:
    """Class representing a node in a document graph."""

    def __init__(
        self,
        children: list["Node"] | None = None,
        metadata: dict | None = None,
        node_type: NodeType = NodeType.SECTION,
        content: str | None = None,
    ):
        """
        Initialize a Node object.

        Args:
            children: List of child nodes
            metadata: Dictionary of metadata
            content: Content object for leaf nodes
        """
        self.children = children or []
        self.metadata = metadata or {}
        self.node_type = node_type
        self._content = content
        self.id = uuid.uuid4()

    @property
    def content(self) -> str:
        """Get the textual content of this node and its children."""
        if self.is_leaf:
            if self._content:
                return self._content
            # Handle other content types appropriately
            return ""

        # Add space or newline between child contents
        return " ".join([child.content for child in self.children])

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node (has no children)."""
        return len(self.children) == 0

    def add_child(self, child: "Node") -> None:
        """Add a child node to this node."""
        self.children.append(child)

    def remove_child(self, child: "Node") -> bool:
        """Remove a child node from this node."""
        if child in self.children:
            self.children.remove(child)
            return True
        return False

    def __eq__(self, other: "Node") -> bool:
        """Check if two nodes are equal."""
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash the node."""
        return hash(self.id)


class MediaNode(Node):
    """Class representing a media node in a document graph."""

    def __init__(
        self,
        media: Media,
        children: list["Node"] | None = None,
        metadata: dict | None = None,
        node_type: NodeType = NodeType.SECTION,
        content: str | None = None,
    ):
        super().__init__(
            children=children,
            metadata=metadata,
            node_type=node_type,
            content=content,
        )
        self.media = media


class EmbeddingFormat(Enum):
    DENSE = "dense"
    SPARSE = "sparse"


@dataclass
class Embedding:
    name: str


@dataclass
class DenseEmbedding(Embedding):
    vector: list[float]


@dataclass
class SparseEmbedding(Embedding):
    values: list[float]
    indices: list[int]


@dataclass
class Chunk:
    """
    A class for storing the chunk response from a client.
    """

    def __init__(
        self,
        id: str,
        text: str,
        embeddings: list[Embedding] | None = None,
        metadata: dict | None = None,
    ):
        """
        Initialize a Chunk object.

        Args:
            id (str): The id of the chunk.
            text (str): The text of the chunk.
            embeddings (list[Embedding], optional): The embeddings of the chunk. Defaults to [].
            metadata (dict, optional): The metadata of the chunk. Defaults to {}.
        """
        self.id = id
        self.text = text
        self.embeddings = embeddings or []
        self.metadata = metadata or {}
