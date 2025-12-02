import re

from datapizza.core.modules.parser import Parser
from datapizza.type import Node, NodeType


class MDParser(Parser):
    """
    Parser that creates a hierarchical tree structure from Markdown file.
    The hierarchy goes from document -> sections -> paragraphs -> sentences.
    """

    def __init__(self):
        """Initialize the MDParser."""
        super().__init__()
        # Regex pattern for splitting text into sentences (same as TextParser)
        self.sentence_pattern = re.compile(
            r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
        )
        # Regex to match markdown headers
        self.header_pattern = re.compile(r"^(#+)\s+(.*)")

    def parse(self, file_path: str, metadata: dict | None = None) -> Node:
        """
        Parse markdown file into a hierarchical tree structure.

        Args:
            file_path: The path to the markdown file to parse
            metadata: Optional metadata for the root node

        Returns:
            A Node representing the document with section, paragraph and sentence nodes
        """
        if metadata is None:
            metadata = {}
        metadata["file_path"] = file_path

        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        document_node = Node(
            children=[], metadata=metadata.copy(), node_type=NodeType.DOCUMENT
        )

        # Stack of (level, node). Document is level 0.
        stack: list[tuple[int, Node]] = [(0, document_node)]

        lines = text.split("\n")
        current_paragraph_lines = []

        def flush_paragraph():
            if not current_paragraph_lines:
                return

            para_text = " ".join(current_paragraph_lines).strip()
            if not para_text:
                return

            # Create paragraph node
            paragraph_node = Node(
                children=[],
                metadata=metadata.copy(),  # Add metadata to paragraph
                node_type=NodeType.PARAGRAPH,
            )

            # Split into sentences
            sentences = self._split_sentences(para_text)
            for i, sentence_text in enumerate(sentences):
                sent_metadata = metadata.copy()
                sent_metadata.update({"index": i, "text": sentence_text})
                sentence_node = Node(
                    children=[],
                    metadata=sent_metadata,
                    node_type=NodeType.SENTENCE,
                    content=sentence_text,
                )
                paragraph_node.add_child(sentence_node)

            # Add paragraph to the current active section (top of stack)
            stack[-1][1].add_child(paragraph_node)
            current_paragraph_lines.clear()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            header_match = self.header_pattern.match(line)
            if header_match:
                # Flush any pending paragraph text before starting new section
                flush_paragraph()

                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Pop stack until we find a parent with lower level
                while stack and stack[-1][0] >= level:
                    stack.pop()

                sec_metadata = metadata.copy()
                sec_metadata.update({"title": title, "level": level})

                # Create new section node
                section_node = Node(
                    children=[], metadata=sec_metadata, node_type=NodeType.SECTION
                )

                # Add to parent
                if stack:
                    stack[-1][1].add_child(section_node)

                # Push to stack
                stack.append((level, section_node))
            else:
                # Accumulate text for paragraph
                current_paragraph_lines.append(line)

        # Flush any remaining paragraph text
        flush_paragraph()

        return document_node

    async def a_parse(self, file_path: str, metadata: dict | None = None) -> Node:
        return self.parse(file_path, metadata)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
