import json
from typing import Any

import aiofiles
from datapizza.core.modules.parser import Parser
from datapizza.core.utils import extract_media
from datapizza.type import Media, MediaNode, Node, NodeType

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.aio import (
    DocumentIntelligenceClient as AsyncDocumentIntelligenceClient,
)
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeResult
from azure.core.credentials import AzureKeyCredential


class AzureParser(Parser):
    """
    Parser that creates a hierarchical tree structure from Azure AI Document Intelligence response.
    The hierarchy goes from document -> pages -> paragraphs/tables -> lines/cells -> words.

    params:
        api_key: str
        endpoint: str
        result_type: str = "markdown", "text"
    """

    def __init__(self, api_key: str, endpoint: str, result_type: str = "text"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.result_type = result_type
        self.parser = None  # self._create_parser()
        self.a_parser = None  # self._create_a_parser()

    def _create_parser(self):
        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=self.endpoint, credential=AzureKeyCredential(self.api_key)
        )
        return document_intelligence_client

    def _create_a_parser(self):
        parser = AsyncDocumentIntelligenceClient(
            endpoint=self.endpoint, credential=AzureKeyCredential(self.api_key)
        )
        return parser

    def _get_parser(self):
        if not self.parser:
            self.parser = self._create_parser()
        return self.parser

    def _get_a_parser(self):
        if not self.a_parser:
            self.a_parser = self._create_a_parser()
        return self.a_parser

    def _parse_file(self, file_path: str) -> Node:
        """Parse an Azure Document Intelligence JSON file into a Node structure."""
        with open(file_path) as file:
            json_data = json.load(file)

        return self._parse_json(json_data, file_path=file_path)

    def _get_missing_paragraphs(self, json_data: dict) -> list[str]:
        """Get missing paragraphs from the Azure Document Intelligence JSON data."""

        sections = json_data.get("sections", [])
        figures = json_data.get("figures", [])
        tables = json_data.get("tables", [])

        all_paragraphs = [
            "/paragraphs/" + str(x) for x in range(len(json_data.get("paragraphs", [])))
        ]

        elements = []

        def _process_section(section):
            for element in section.get("elements", []):
                if "paragraph" in element:
                    elements.append(element)
                elif "section" in element:
                    section_idx = element.split("/")[2]
                    next_section = sections[int(section_idx)]
                    _process_section(next_section)

        for section in sections:
            _process_section(section)

        def _process_figure(figure):
            for element in figure.get("elements", []):
                if "paragraph" in element:
                    elements.append(element)
                elif "section" in element:
                    section_idx = element.split("/")[2]
                    next_section = sections[int(section_idx)]
                    _process_section(next_section)

        for figure in figures:
            _process_figure(figure)

        def _process_table(table):
            for element in table.get("elements", []):
                if "paragraph" in element:
                    elements.append(element)
                elif "section" in element:
                    section_idx = element.split("/")[2]
                    next_section = sections[int(section_idx)]
                    _process_section(next_section)

        for table in tables:
            _process_table(table)

        missing = [x for x in all_paragraphs if x not in elements]

        return missing

    def _insert_missing_paragraphs(self, json_data: dict) -> dict:
        """Insert missing paragraphs into the Azure Document Intelligence JSON data."""

        missing = self._get_missing_paragraphs(json_data)

        def _insert_paragraph_recursive(section, p_idx, p):
            for i, element in enumerate(section.get("elements", [])):
                if "paragraph" in element:
                    if int(element.split("/")[2]) > int(p_idx):
                        section["elements"].insert(i, p)
                        return True
                elif "section" in element:
                    section_idx = element.split("/")[2]
                    next_section = json_data["sections"][int(section_idx)]
                    if _insert_paragraph_recursive(next_section, p_idx, p):
                        return True
            return False

        for p in missing:
            idx = int(p.split("/")[2])

            for section in json_data.get("sections", []):
                if _insert_paragraph_recursive(section, idx, p):
                    break
        return json_data

    def _parse_json(self, json_data: dict, file_path: str) -> Node:
        """
        Parse Azure Document Intelligence JSON into a hierarchical Node structure.

        Args:
            json_data: The Azure Document Intelligence JSON response

        Returns:
            A Node representing the document with hierarchical structure
        """
        # Create root document node

        json_data = self._insert_missing_paragraphs(json_data)

        document_node = Node(
            children=[],
            metadata=self._extract_document_metadata(json_data),
            node_type=NodeType.DOCUMENT,
        )

        # Process each page in the document
        analyze_result = json_data  # .get('analyzeResult', {})
        sections = analyze_result.get("sections", [])

        document_node.children = self._process_children_elements(
            sections[0], analyze_result, file_path=file_path
        )

        return document_node

    def _process_children_elements(
        self,
        parent_object: dict[str, Any],
        analyze_result: dict[str, Any],
        file_path: str,
    ) -> list[Node]:
        """Process children elements of a section."""
        children_nodes = []
        elements = parent_object.get("elements", [])
        for _element_idx, element in enumerate(elements):
            if "paragraph" in element:
                paragrap_index = element.split("/")[2]

                paragraph = analyze_result.get("paragraphs", [])[int(paragrap_index)]
                paragraph_node = self._create_paragraph_node(paragraph)
                paragraph_node.children = self._process_children_elements(
                    paragraph, analyze_result, file_path=file_path
                )
                children_nodes.append(paragraph_node)

            elif "table" in element:
                table_index = element.split("/")[2]
                table = analyze_result.get("tables", [])[int(table_index)]
                table_node = self._create_media_node(
                    media=table,
                    node_type=NodeType.TABLE,
                    content_result=analyze_result.get("content", ""),
                    file_path=file_path,
                )
                table_node.children = self._process_children_elements(
                    table, analyze_result, file_path=file_path
                )
                children_nodes.append(table_node)

            elif "figures" in element:
                image_index = element.split("/")[2]
                image = analyze_result.get("figures", [])[int(image_index)]
                image_node = self._create_media_node(
                    media=image,
                    node_type=NodeType.FIGURE,
                    content_result=analyze_result.get("content", ""),
                    file_path=file_path,
                )
                image_node.children = self._process_children_elements(
                    image, analyze_result, file_path=file_path
                )
                children_nodes.append(image_node)

            elif "section" in element:
                section_index = element.split("/")[2]
                section = analyze_result.get("sections", [])[int(section_index)]
                section_node = Node(children=[], node_type=NodeType.SECTION)
                section_node.children = self._process_children_elements(
                    section, analyze_result, file_path=file_path
                )
                children_nodes.append(section_node)

        return children_nodes

    def _transform_cells_to_markdown(
        self, table_data: dict[str, Any], content_result: str
    ) -> str:
        """Transforms table cells from Azure response to a markdown table string."""
        cells = table_data.get("cells", [])
        if not cells:
            return ""

        offset = table_data.get("spans", [{}])[0].get("offset")
        length = table_data.get("spans", [{}])[0].get("length")
        if offset is None or length is None:
            return ""

        markdown_table = content_result[offset : offset + length]

        return markdown_table

    def _create_media_node(
        self,
        media: dict[str, Any],
        node_type: NodeType,
        content_result: str,
        file_path: str,
    ) -> Node:
        """Create a node for an media with its child elements."""
        # Get bounding regions
        bounding_regions = media.get("boundingRegions", [])

        if file_path and bounding_regions:
            base64_image = extract_media(
                coordinates=bounding_regions[0]["polygon"],
                file_path=file_path,
                page_number=bounding_regions[0]["pageNumber"],
            )

            media_obj = Media(
                media_type="image",
                source=base64_image,
                source_type="base64",
                extension="png",
            )
        else:
            raise ValueError("No bounding regions found for media")

        content = None
        metadata = {
            "boundingRegions": bounding_regions,
        }
        if node_type == NodeType.TABLE:
            content = self._transform_cells_to_markdown(media, content_result)
            metadata["rowCount"] = media.get("rowCount")
            metadata["columnCount"] = media.get("columnCount")

        # Create MediaNode with bounding regions metadata
        image_node = MediaNode(
            media=media_obj,
            children=[],
            node_type=node_type,
            metadata=metadata,
            content=content,
        )
        return image_node

    def _extract_document_metadata(self, json_data: dict[str, Any]) -> dict[str, Any]:
        """Extract document-level metadata from the Azure response."""
        metadata = {}
        analyze_result = json_data.get("analyzeResult", {})

        # Add document-level metadata
        if "documentResults" in analyze_result:
            doc_results = analyze_result["documentResults"]
            if doc_results and len(doc_results) > 0:
                metadata.update(doc_results[0].get("fields", {}))

        # Add model information if available
        metadata["modelId"] = analyze_result.get("modelId")
        metadata["apiVersion"] = analyze_result.get("apiVersion")

        return metadata

    # def _create_table_node(self, table: Dict[str, Any]) -> Node:
    #     """Create a node for a table with its child lines and words."""
    #     table_node = Node(
    #         children=[],
    #         node_type=NodeType.TABLE,
    #         content=table.get("content", ""),
    #         metadata={
    #             "boundingRegions": table.get("boundingRegions", []),
    #         },
    #     )
    #     return table_node

    def _create_paragraph_node(self, paragraph: dict[str, Any]) -> Node:
        """Create a node for a paragraph with its child lines and words."""
        para_node = Node(
            children=[],
            node_type=NodeType.PARAGRAPH,
            content=paragraph.get("content", ""),
            metadata={
                "boundingRegions": paragraph.get("boundingRegions", {}),
            },
        )
        return para_node

    def parse_with_azure_ai(self, file_path: str) -> dict:
        """
        Parse a Document with Azure AI Document Intelligence into a json dictionary.

        Args:
            file_path: Path to the document

        Returns:
            A dictionary with the Azure AI Document Intelligence response
        """

        with open(file_path, "rb") as file:
            file_content = file.read()

        parser = self._get_parser()
        poller = parser.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=file_content),
            output_content_format=self.result_type,
        )
        result: AnalyzeResult = poller.result()
        return result.as_dict()

    async def a_parse_with_azure_ai(self, file_path: str) -> dict:
        async with aiofiles.open(file_path, "rb") as file:
            file_content = await file.read()

        parser = self._get_a_parser()
        async with parser:
            poller = await parser.begin_analyze_document(
                "prebuilt-layout",
                AnalyzeDocumentRequest(bytes_source=file_content),
                output_content_format=self.result_type,
            )
            result: AnalyzeResult = await poller.result()
            return result.as_dict()

    def parse(self, file_path: str, metadata: dict | None = None) -> Node:
        """
        Parse a Document with Azure AI Document Intelligence into a Node
        structure.

        Args:
            file_path: Path to the document
            metadata: Optional metadata to be merged into the root document
                node. Defaults to None.

        Returns:
            A Node representing the document with hierarchical structure

        Raises:
            TypeError: If metadata is not a dict or None
        """
        # Validate metadata type
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(f"metadata must be a dict or None, got {type(metadata).__name__}")

        result_dict = self.parse_with_azure_ai(file_path)
        document_node = self._parse_json(result_dict, file_path=file_path)

        # Merge provided metadata into the document node's metadata
        if metadata:
            document_node.metadata.update(metadata)

        return document_node

    def __call__(self, file_path: str, metadata: dict | None = None) -> Node:
        """
        Allow the parser to be called directly as a function.

        Args:
            file_path: Path to the document
            metadata: Optional metadata to be merged into the root document node

        Returns:
            A Node representing the document with hierarchical structure
        """
        return self.parse(file_path, metadata)

    async def a_parse(
        self, file_path: str, metadata: dict | None = None
    ) -> Node:
        """
        Async version of parse().

        Args:
            file_path: Path to the document
            metadata: Optional metadata to be merged into the root document
                node. Defaults to None.

        Returns:
            A Node representing the document with hierarchical structure

        Raises:
            TypeError: If metadata is not a dict or None
        """
        # Validate metadata type
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(f"metadata must be a dict or None, got {type(metadata).__name__}")

        result_dict = await self.a_parse_with_azure_ai(file_path)
        document_node = self._parse_json(result_dict, file_path=file_path)

        # Merge provided metadata into the document node's metadata
        if metadata:
            document_node.metadata.update(metadata)

        return document_node
