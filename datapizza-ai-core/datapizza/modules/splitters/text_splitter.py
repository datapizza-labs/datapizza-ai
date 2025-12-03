import uuid

from datapizza.core.modules.splitter import Splitter
from datapizza.type.type import Chunk


class TextSplitter(Splitter):
    """
    A basic text splitter that operates directly on strings rather than Node objects.
    Unlike other splitters that work with Node types, this splitter takes raw text input
    and splits it into chunks while maintaining configurable size and overlap parameters.

    """

    def __init__(self, max_char: int = 5000, overlap: int = 0):
        """
        Initialize the TextSplitter.

        Args:
            max_char: The maximum number of characters per chunk
            overlap: The number of characters to overlap between chunks
        """

        self.max_char = max_char
        self.overlap = overlap

    def split(self, text: str) -> list[Chunk]:
        """
        Split the text into chunks.

        Args:
            text: The text to split

        Returns:
            A list of chunks
        """
        if not isinstance(text, str):
            raise TypeError("TextSplitter expects a string input")

        text_length = len(text)
        if text_length == 0:
            return []

        if text_length <= self.max_char:
            return [Chunk(id=str(uuid.uuid4()), text=text, metadata={})]

        # Ensure progress even if overlap is large
        step = max(1, self.max_char - max(0, self.overlap))

        chunks: list[Chunk] = []
        start = 0
        while start < text_length:
            end = min(start + self.max_char, text_length)
            chunk_text = text[start:end]
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    metadata={"start_char": start, "end_char": end},
                )
            )

            if end >= text_length:
                break
            start += step

        return chunks

    async def a_split(self, text: str) -> list[Chunk]:
        return self.split(text)
