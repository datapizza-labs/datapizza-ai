import re
import uuid

from datapizza.core.modules.splitter import Splitter
from datapizza.type.type import Chunk


class TextSplitter(Splitter):
    # Delimiter patterns for different split levels
    WORD_PATTERN = r"\s+"  # Split on whitespace
    PHRASE_PATTERN = r"(?<=[.!?])\s+"  # Split on sentence boundaries
    PARAGRAPH_PATTERN = r"\n\n+|\r\n\r\n+|\r\r+"  # Split on paragraph breaks
    """
    A basic text splitter that operates directly on strings rather than Node objects.
    Unlike other splitters that work with Node types, this splitter takes raw text input
    and splits it into chunks while maintaining configurable size and overlap parameters.

    """

    def __init__(
        self,
        max_char: int = 5000,
        overlap: int = 0,
        split_level: str = "char",
        min_overlap_words: int = 1,
    ):
        """
        Initialize the TextSplitter.

        Args:
            max_char: The maximum number of characters per chunk
            overlap: The number of characters/words to overlap between chunks
            split_level: The level at which to split text. Options: "char", "word", "phrase", "paragraph"
            min_overlap_words: Minimum number of words required for word-based overlap
        """
        if split_level not in ("char", "word", "phrase", "paragraph"):
            raise ValueError(
                f"Invalid split_level: {split_level}. Must be one of: char, word, phrase, paragraph"
            )

        self.max_char = max_char
        self.overlap = overlap
        self.split_level = split_level
        self.min_overlap_words = min_overlap_words
        self.patterns = {
            "word": self.WORD_PATTERN,
            "phrase": self.PHRASE_PATTERN,
            "paragraph": self.PARAGRAPH_PATTERN,
        }

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

        # Route to appropriate splitting method
        if self.split_level == "char":
            return self._split_char(text)

        return self._split_with_delimiter(text, self.split_level)

    def _split_char(self, text: str) -> list[Chunk]:
        """
        Split the text by character count.

        Args:
            text: The text to split

        Returns:
            A list of chunks
        """
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
            chunks.append(Chunk(id=str(uuid.uuid4()), text=chunk_text, metadata={}))

            if end >= text_length:
                break
            start += step

        return chunks

    def _split_with_delimiter(
            self,
            text: str,
            level: str,
            pattern: str = None,
            skip_overlap=False
    ) -> list[Chunk]:
        """
        Split text using a delimiter pattern while respecting max_char limits.

        Segments are combined into chunks until the step limit is reached (max_char - overlap).
        Oversized segments are handled by falling back to word-level splitting, or character-level
        if already at word level.

        Args:
            text: The text to split
            level: The current split level (for overflow handling)
            pattern: The regex pattern to split on (optional, derived from level if not provided)
            skip_overlap: If True, skip applying overlap to the resulting chunks (default: False)

        Returns:
            A list of chunks
        """
        # may be no default provided, so fail if wrong
        pattern = pattern if pattern else self.patterns.get(level, self.WORD_PATTERN)

        # Split text by delimiter (delimiters are discarded)
        segments = re.split(pattern, text)
        segments = [s for s in segments if s]  # Remove empty segments

        if not segments:
            return []

        # Reserve space for overlap if needed (proactive approach)
        step = max(1, self.max_char - max(0, self.overlap))
        chunks: list[Chunk] = []
        current_chunk = ""

        for segment in segments:

            # Try adding this segment to current chunk
            test_chunk = current_chunk +  " " + segment if current_chunk else segment

            if len(test_chunk) <= step:
                current_chunk = test_chunk
                continue

            # Current chunk is full, add to results
            if current_chunk:
                chunks.append(
                    Chunk(id=str(uuid.uuid4()), text=current_chunk, metadata={})
                )

            # Handle the new segment
            if len(segment) <= step:
                current_chunk = segment
                continue

            # Segment is too large, needs overflow handling
            overflow_chunks = self._handle_overflow(segment, level)
            chunks.extend(overflow_chunks)
            current_chunk = ""

        # Add remaining chunk
        if current_chunk:
            chunks.append(Chunk(id=str(uuid.uuid4()), text=current_chunk, metadata={}))

        # Apply overlap if needed
        if not skip_overlap and self.overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)



        return chunks

    def _apply_overlap(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Apply overlap between chunks by appending text from the next chunk to the current chunk.

        Args:
            chunks: List of chunks to apply overlap to

        Returns:
            List of chunks with overlap applied
        """
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks: list[Chunk] = []

        for i, chunk in enumerate(chunks):

            if i == len(chunks) - 1:
                # Last chunk has no suffix overlap
                overlapped_chunks.append(chunk)
                continue

            # Get overlap text from next chunk (beginning of next chunk)
            next_chunk = chunks[i + 1]
            # Calculate available space for overlap counting also the separator space
            available_space = max(0, self.max_char - len(chunk.text) - 1)
            overlap_text = self._get_starting_text(next_chunk.text, available_space)

            # Combine current chunk with overlap from next
            new_text = chunk.text + " " + overlap_text
            overlapped_chunks.append(
                Chunk(id=str(uuid.uuid4()), text=new_text, metadata=chunk.metadata)
            )

        return overlapped_chunks

    def _get_starting_text(self, text: str,current_overlap: int) -> str:
        """
        Extract overlap text from the beginning of a chunk.

        Strategy:
        1. If the first current_overlap characters don't contain enough words (< min_overlap_words),
           return character-based overlap (first current_overlap characters).
        2. Otherwise, find the maximum number of complete words that fit within current_overlap characters,
           ensuring we have at least min_overlap_words.

        Args:
            text: The text to extract overlap from
            current_overlap: The maximum number of characters to use for overlap

        Returns:
            The overlap text to append to the previous chunk
        """

        # Check if we have enough words in the first current_overlap characters
        words_in_overlap = re.split(self.WORD_PATTERN, text[:current_overlap])
        if len(words_in_overlap) <= self.min_overlap_words:
            # Not enough words, use character-based overlap
            return text[:current_overlap]

        # We have enough words, find how many complete words fit in current_overlap chars
        words = re.split(self.WORD_PATTERN, text)
        candidate = ""
        for pos in range(0, len(words)):
            if len(" ".join(words[:pos])) > current_overlap:
                break
            candidate = " ".join(words[:pos])

        return candidate


    def _handle_overflow(self, text: str, current_level: str) -> list[Chunk]:
        """
        Handle text segments that exceed max_char.

        Fallback strategy: word level falls back to char splitting,
        all other levels fall back to word splitting.

        Args:
            text: The oversized text segment
            current_level: The current split level

        Returns:
            List of chunks created from the oversized segment
        """
        if current_level != "word":
            return self._split_with_delimiter(text, "word",skip_overlap=True)

        return self._split_char(text)

    async def a_split(self, text: str) -> list[Chunk]:
        return self.split(text)
