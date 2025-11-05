import json
import warnings
from pathlib import Path
from typing import Any

from datapizza.core.modules.parser import Parser
from datapizza.type import Node, NodeType
from datapizza.type.type import Media, MediaNode

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from .ocr_options import OCROptions
from .utils import extract_media_from_docling_bbox


class DoclingParser(Parser):
    """
    Parser that converts files using Docling and then converts the resulting
    DoclingDocument JSON into a datapizza Node tree.
    Supported file extensions: https://docling-project.github.io/docling/usage/supported_formats/

    - Accepts files directly and processes them using Docling DocumentConverter
    - Logical-only hierarchy (no page nodes)
    - Paragraphs are leaf nodes; no sentence splitting by default
    - Full preservation of Docling items stored in node.metadata["docling_raw"],
      with convenience fields (docling_type, docling_label, bbox, page_no, self_ref)
    - Reading order follows body.children ($ref list)
    - Images and tables are mapped to FIGURE and TABLE nodes respectively, with bbox-only metadata
    """
    def __init__(
        self,
        json_output_dir: str | None = None,
        ocr_options: OCROptions | None = None,
    ):
        """
        Initialize DoclingParser with optional OCR configuration.

        Args:
            json_output_dir: Optional directory to save intermediate Docling JSON results.
            ocr_options: OCR configuration. Defaults to EasyOCR (backward compatible) if not provided.
        """
        self.converter = None
        # Optional directory to save intermediate Docling JSON results
        self.json_output_dir = json_output_dir
        self.ocr_options = ocr_options or OCROptions()

    def _create_converter(self):
        """
        Create a Docling DocumentConverter with configured OCR options.

        Uses self.ocr_options to determine which OCR engine to use and
        passes the appropriate configuration to Docling.

        Returns:
            Configured DocumentConverter instance
        """
        # Convert internal OCR options to Docling pipeline kwargs
        pipeline_kwargs = self.ocr_options.to_docling_pipeline_options()
        pipeline_options = PdfPipelineOptions(**pipeline_kwargs)

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            },
        )

    def _get_converter(self):
        """Get or create converter instance."""
        if not self.converter:
            self.converter = self._create_converter()
        return self.converter

    def _is_json_file(self, file_path: str) -> bool:
        """Check if the file is a JSON file."""
        return Path(file_path).suffix.lower() == ".json"

    def parse_to_json(self, file_path: str) -> dict:
        """
        Parse a supported file using Docling, or if json_path is provided, load that
        Docling JSON directly and skip conversion.

        Args:
            file_path: Path to the source File (required for media extraction)
            json_path: Optional path to a Docling JSON file to skip conversion

        Returns:
            Docling document as a dictionary
        """

        converter = self._get_converter()
        result = converter.convert(file_path)
        doc_dict = result.document.export_to_dict()

        # Optionally persist intermediate Docling JSON beside the pipeline output
        if self.json_output_dir:
            out_dir = Path(self.json_output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{Path(file_path).stem}.json"
            with open(out_path, "w", encoding="utf-8") as fp:
                json.dump(doc_dict, fp, ensure_ascii=False, indent=2)

        return doc_dict

    def _json_to_node(self, json_data: dict, file_path: str | None = None) -> Node:
        """
        Convert Docling JSON into a Node hierarchy.
        """
        # Root document with full raw preservation
        document_node = Node(children=[], node_type=NodeType.DOCUMENT, metadata={})
        # Preserve entire Docling raw at root for losslessness
        document_node.metadata["docling_raw"] = json_data
        document_node.metadata["schema_name"] = json_data.get("schema_name")
        document_node.metadata["version"] = json_data.get("version")
        document_node.metadata["name"] = json_data.get("name")
        document_node.metadata["origin"] = json_data.get("origin")

        # Prepare ref resolver over top-level collections
        collections: dict[str, dict[str, Any]] = {}
        for key, value in json_data.items():
            if isinstance(value, list):
                collections[key] = {
                    f"#/{key}/{i}": item for i, item in enumerate(value)
                }

        def resolve_ref(ref_obj: Any) -> tuple[str, dict[str, Any]] | None:
            """
            Resolve a Docling $ref object or a direct item.

            Returns (collection_name, item_dict) when resolved from ref, or
            (derived_type, item_dict) when passed a direct item.
            """
            if isinstance(ref_obj, dict) and "$ref" in ref_obj:
                ref_path = ref_obj["$ref"]
                # find collection that has this ref
                for coll_name, index_map in collections.items():
                    if ref_path in index_map:
                        return coll_name, index_map[ref_path]
                return None
            elif isinstance(ref_obj, dict) and "self_ref" in ref_obj:
                # Direct item already expanded
                # Infer collection from self_ref prefix if present
                self_ref: str = ref_obj.get("self_ref", "")
                coll_guess = (
                    self_ref.split("/")[1] if self_ref.startswith("#/") else "unknown"
                )
                return coll_guess, ref_obj
            else:
                return None

        # Reading order source
        body: dict[str, Any] = json_data.get("body", {})
        body_children: list[Any] = body.get("children", [])

        # Maintain a stack of open sections based on level
        section_stack: list[tuple[int, Node]] = []  # list of (level, node)
        # Accumulate consecutive list items for markdown list rendering
        current_list_items: list[dict[str, Any]] = []

        def current_parent() -> Node:
            return section_stack[-1][1] if section_stack else document_node

        def push_section(
            level: int, header_item: dict[str, Any], coll_name: str
        ) -> Node:
            # Pop deeper/equal levels
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()

            section_node = Node(children=[], node_type=NodeType.SECTION, metadata={})
            # Preserve header as metadata
            section_node.metadata["docling_raw"] = header_item
            section_node.metadata["docling_type"] = coll_name
            section_node.metadata["docling_label"] = header_item.get("label")
            section_node.metadata["level"] = header_item.get("level")
            _add_common_metadata(section_node.metadata, header_item)

            parent = current_parent()
            parent.add_child(section_node)
            section_stack.append((level, section_node))

            # Also emit a heading paragraph as markdown line under the section
            heading_text: str = header_item.get("text", "") or ""
            # Bump heading level by 1 to match expected markdown (level 1 -> '##')
            hlevel = max(1, min(6, int(level or 1) + 1))
            heading_md: str = f"{'#' * hlevel} {heading_text}".rstrip()
            if heading_md.strip():
                heading_para = Node(
                    children=[],
                    node_type=NodeType.PARAGRAPH,
                    metadata={},
                    content=heading_md + "\n\n",
                )
                # Keep original header in paragraph metadata too for direct association
                heading_para.metadata["docling_header_ref"] = header_item.get(
                    "self_ref"
                )
                heading_para.metadata["markdown_rendering"] = "heading"
                # Also carry over full Docling metadata so headers are discoverable as nodes
                heading_para.metadata["docling_raw"] = header_item
                heading_para.metadata["docling_type"] = coll_name
                heading_para.metadata["docling_label"] = header_item.get("label")
                heading_para.metadata["level"] = header_item.get("level")
                _add_common_metadata(heading_para.metadata, header_item)
                section_node.add_child(heading_para)
            return section_node

        def _normalize_coord_origin(origin_val: Any) -> str:
            """Normalize coord origin to either 'BOTTOMLEFT' or 'TOPLEFT'."""
            if hasattr(origin_val, "name"):
                origin_str = str(getattr(origin_val, "name", "")).upper()
            else:
                origin_str = str(origin_val).upper()
            if "." in origin_str:
                origin_str = origin_str.split(".")[-1]
            origin_str = origin_str.replace("_", "").replace("-", "")
            if origin_str not in {"BOTTOMLEFT", "TOPLEFT"}:
                # Default to Docling default
                origin_str = "BOTTOMLEFT"
            return origin_str

        def _union_two_bboxes(b1: dict[str, Any], b2: dict[str, Any]) -> dict[str, Any]:
            """Return the union bbox of two Docling-style rects, honoring coord origin.

            Expects keys l,t,r,b and optional coord_origin (defaults to BOTTOMLEFT).
            The resulting bbox uses the origin of b1 if present, otherwise b2.
            """
            origin = _normalize_coord_origin(
                b1.get("coord_origin", b2.get("coord_origin", "BOTTOMLEFT"))
            )
            l1, r1 = float(b1.get("l", 0.0)), float(b1.get("r", 0.0))
            t1, btm1 = float(b1.get("t", 0.0)), float(b1.get("b", 0.0))
            l2, r2 = float(b2.get("l", 0.0)), float(b2.get("r", 0.0))
            t2, btm2 = float(b2.get("t", 0.0)), float(b2.get("b", 0.0))

            left = min(l1, l2)
            right = max(r1, r2)
            if origin == "BOTTOMLEFT":
                top = max(t1, t2)
                bottom = min(btm1, btm2)
            else:  # TOPLEFT
                top = min(t1, t2)
                bottom = max(btm1, btm2)

            return {
                "l": left,
                "t": top,
                "r": right,
                "b": bottom,
                "coord_origin": origin,
            }

        def _merge_bboxes_per_page(
            prov_list: list[dict[str, Any]],
        ) -> dict[int, dict[str, Any]]:
            """Merge multiple bbox entries by page into a single union per page.

            Returns a mapping {page_no: merged_bbox}.
            Ignores malformed entries that lack page_no or bbox.
            """
            page_to_bbox: dict[int, dict[str, Any]] = {}
            for prov in prov_list:
                try:
                    page_no = int((prov or {}).get("page_no") or 0)
                except Exception:
                    page_no = 0
                bbox = (prov or {}).get("bbox") or None
                if page_no <= 0 or not isinstance(bbox, dict):
                    continue
                if page_no not in page_to_bbox:
                    page_to_bbox[page_no] = bbox
                else:
                    page_to_bbox[page_no] = _union_two_bboxes(
                        page_to_bbox[page_no], bbox
                    )
            return page_to_bbox

        def _add_common_metadata(
            metadata: dict[str, Any], item: dict[str, Any]
        ) -> None:
            metadata["self_ref"] = item.get("self_ref")
            metadata["prov"] = item.get("prov")
            # Merge multi-bbox per page; keep a representative bbox for the first page
            if isinstance(item.get("prov"), list) and len(item["prov"]) > 0:
                merged = _merge_bboxes_per_page(item["prov"])  # {page_no: bbox}
                if merged:
                    page_nos_sorted = sorted(merged.keys())
                    first_page = page_nos_sorted[0]
                    metadata["page_no"] = first_page
                    metadata["bbox"] = merged[first_page]
                    if len(merged) > 1:
                        metadata["page_nos"] = page_nos_sorted
                        metadata["bbox_per_page"] = merged
                    # Expose Azure-style boundingRegions so downstream splitter/vectorstore receive them
                    # We approximate polygons as rectangles from Docling bbox; values expressed in inches
                    bounding_regions: list[dict[str, Any]] = []
                    for pg in page_nos_sorted:
                        bb = merged.get(pg) or {}
                        try:
                            l = float(bb.get("l", 0.0)) / 72.0
                            r = float(bb.get("r", 0.0)) / 72.0
                            t = float(bb.get("t", 0.0)) / 72.0
                            btm = float(bb.get("b", 0.0)) / 72.0
                        except Exception:
                            # Skip malformed bbox
                            continue
                        polygon = [
                            l,
                            btm,
                            r,
                            btm,
                            r,
                            t,
                            l,
                            t,
                        ]
                        bounding_regions.append(
                            {
                                "pageNumber": int(pg),
                                "polygon": polygon,
                                # Preserve original coord origin for consumers that need it
                                "coordOrigin": (bb.get("coord_origin") or "BOTTOMLEFT"),
                            }
                        )
                    if bounding_regions:
                        metadata["boundingRegions"] = bounding_regions
            # pass through common fields
            for k in ("label", "content_layer", "orig", "text"):
                if k in item:
                    metadata[k] = item.get(k)

        def _create_paragraph_with_sentences(
            text_item: dict[str, Any], coll_name: str
        ) -> Node:
            paragraph_node = Node(
                children=[], node_type=NodeType.PARAGRAPH, metadata={}
            )
            paragraph_node.metadata["docling_raw"] = text_item
            paragraph_node.metadata["docling_type"] = coll_name
            paragraph_node.metadata["docling_label"] = text_item.get("label")
            _add_common_metadata(paragraph_node.metadata, text_item)

            text_content: str = text_item.get("text", "") or ""
            # Store full paragraph text without arbitrary sentence splitting
            # Ensure separation with trailing blank line
            paragraph_node._content = (text_content + "\n\n") if text_content else ""

            return paragraph_node

        def _flush_current_list(parent: Node) -> None:
            nonlocal current_list_items
            if not current_list_items:
                return
            # Create a list parent and one child per bullet item
            list_parent = Node(children=[], node_type=NodeType.PARAGRAPH, metadata={})
            list_parent.metadata["markdown_rendering"] = "list"
            # Preserve full raw items for losslessness
            list_parent.metadata["docling_list_items_raw"] = current_list_items

            num_items = len(current_list_items)
            for idx, li in enumerate(current_list_items):
                li_text: str = li.get("text", "") or ""
                bullet_node = Node(
                    children=[], node_type=NodeType.PARAGRAPH, metadata={}
                )
                bullet_node.metadata["markdown_rendering"] = "list_item"
                bullet_node.metadata["docling_raw"] = li
                bullet_node.metadata["docling_type"] = "texts"
                bullet_node.metadata["docling_label"] = "list_item"
                _add_common_metadata(bullet_node.metadata, li)
                # Ensure separation: one newline per item, two newlines after the last one
                trailing_newlines = "\n\n" if idx == num_items - 1 else "\n"
                bullet_node._content = (
                    f"- {li_text}" if li_text else "-"
                ) + trailing_newlines
                list_parent.add_child(bullet_node)

            parent.add_child(list_parent)
            current_list_items = []

        def _extract_base64_from_pdf_bbox(prov_entry: dict[str, Any]) -> str | None:
            if not file_path or Path(file_path).suffix.lower() != ".pdf":
                return None
            page_no = int((prov_entry or {}).get("page_no") or 0)
            bbox = (prov_entry or {}).get("bbox") or {}
            if page_no <= 0 or not isinstance(bbox, dict):
                return None
            try:
                return extract_media_from_docling_bbox(
                    bbox=bbox, file_path=file_path, page_number=page_no
                )
            except Exception:
                return None

        def _create_figure_node(item: dict[str, Any], coll_name: str) -> Node:
            media_b64: str | None = None
            prov_list = item.get("prov") or []
            if isinstance(prov_list, list) and len(prov_list) > 0:
                merged = _merge_bboxes_per_page(prov_list)
                if merged:
                    first_page = sorted(merged.keys())[0]
                    media_b64 = _extract_base64_from_pdf_bbox(
                        {"page_no": first_page, "bbox": merged[first_page]}
                    )

            if media_b64:
                media = Media(
                    media_type="image",
                    source_type="base64",
                    source=media_b64,
                    extension="png",
                )
                figure_node: Node = MediaNode(
                    media=media,
                    children=[],
                    node_type=NodeType.FIGURE,
                    metadata={},
                )
            else:
                figure_node = Node(children=[], node_type=NodeType.FIGURE, metadata={})

            figure_node.metadata["docling_raw"] = item
            figure_node.metadata["docling_type"] = coll_name
            figure_node.metadata["docling_label"] = item.get("label", "picture")
            _add_common_metadata(figure_node.metadata, item)
            return figure_node

        def _create_table_node(item: dict[str, Any], coll_name: str) -> Node:
            # Build markdown table content from Docling table data when possible
            content_md: str | None = None
            data = item.get("data", {}) or {}
            table_cells: list[dict[str, Any]] = data.get("table_cells", []) or []
            num_rows: int = int(data.get("num_rows") or 0)
            num_cols: int = int(data.get("num_cols") or 0)

            if table_cells and num_rows > 0 and num_cols > 0:
                # Construct a 2D grid of cell strings, best-effort for simple (no span) cases
                grid: list[list[str]] = [[""] * num_cols for _ in range(num_rows)]
                for cell in table_cells:
                    r0 = int(cell.get("start_row_offset_idx") or 0)
                    c0 = int(cell.get("start_col_offset_idx") or 0)
                    txt = str(cell.get("text") or "")
                    # Prefer first fill; ignore spans for markdown simplicity
                    if 0 <= r0 < num_rows and 0 <= c0 < num_cols and grid[r0][c0] == "":
                        grid[r0][c0] = txt

                # Build markdown lines
                header = grid[0] if num_rows > 0 else []
                # If header is empty, we still create a minimal table
                if header:
                    header_line = (
                        "| " + " | ".join([_escape_md(x) for x in header]) + " |"
                    )
                    sep_line = "| " + " | ".join(["---"] * len(header)) + " |"
                    body_lines = []
                    for r in range(1, num_rows):
                        row_line = (
                            "| " + " | ".join([_escape_md(x) for x in grid[r]]) + " |"
                        )
                        body_lines.append(row_line)
                    content_md = "\n".join([header_line, sep_line, *body_lines])
                else:
                    # Fallback single row from first non-empty row
                    for r in range(num_rows):
                        if any(grid[r]):
                            header = grid[r]
                            header_line = (
                                "| "
                                + " | ".join([_escape_md(x) for x in header])
                                + " |"
                            )
                            sep_line = "| " + " | ".join(["---"] * len(header)) + " |"
                            content_md = "\n".join([header_line, sep_line])
                            break

            if content_md is not None:
                content_md = content_md + "\n\n"

            media_b64: str | None = None
            prov_list = item.get("prov") or []
            if isinstance(prov_list, list) and len(prov_list) > 0:
                merged = _merge_bboxes_per_page(prov_list)
                if merged:
                    first_page = sorted(merged.keys())[0]
                    media_b64 = _extract_base64_from_pdf_bbox(
                        {"page_no": first_page, "bbox": merged[first_page]}
                    )

            if media_b64:
                media = Media(
                    media_type="image",
                    source_type="base64",
                    source=media_b64,
                    extension="png",
                )
                table_node: Node = MediaNode(
                    media=media,
                    children=[],
                    node_type=NodeType.TABLE,
                    metadata={},
                    content=content_md,
                )
            else:
                table_node = Node(
                    children=[],
                    node_type=NodeType.TABLE,
                    metadata={},
                    content=content_md,
                )
            table_node.metadata["docling_raw"] = item
            table_node.metadata["docling_type"] = coll_name
            table_node.metadata["docling_label"] = item.get("label", "table")
            _add_common_metadata(table_node.metadata, item)
            return table_node

        def _append_children_to_media(media_node: Node, item: dict[str, Any]) -> None:
            """
            Append Docling child references (children/captions) directly to the given
            media node (figure/table), deduplicating ref order.
            """
            merged_ref_paths: list[str] = []
            for key in ("children", "captions"):
                seq = item.get(key) or []
                for ref_obj in seq:
                    if isinstance(ref_obj, dict) and "$ref" in ref_obj:
                        merged_ref_paths.append(ref_obj["$ref"])
                    elif isinstance(ref_obj, dict) and ref_obj.get("self_ref"):
                        merged_ref_paths.append(ref_obj["self_ref"])  # unlikely here

            seen: set[str] = set()
            ordered_unique_refs: list[str] = []
            for ref_path in merged_ref_paths:
                if ref_path not in seen:
                    seen.add(ref_path)
                    ordered_unique_refs.append(ref_path)

            for ref_path in ordered_unique_refs:
                resolved_child = resolve_ref({"$ref": ref_path})
                if resolved_child is None:
                    continue
                child_coll, child_item = resolved_child
                child_label = child_item.get("label")

                if child_coll == "texts":
                    node = _create_paragraph_with_sentences(child_item, child_coll)
                    media_node.add_child(node)
                    continue
                if child_coll == "pictures":
                    node = _create_figure_node(child_item, child_coll)
                    # recursively attach nested children if present
                    if (child_item.get("children") or []) or (
                        child_item.get("captions") or []
                    ):
                        _append_children_to_media(node, child_item)
                    media_node.add_child(node)
                    continue
                if child_coll == "tables":
                    node = _create_table_node(child_item, child_coll)
                    if (child_item.get("children") or []) or (
                        child_item.get("captions") or []
                    ):
                        _append_children_to_media(node, child_item)
                    media_node.add_child(node)
                    continue

                # Fallback for unknown child types
                fallback = Node(children=[], node_type=NodeType.PARAGRAPH, metadata={})
                fallback.metadata["docling_raw"] = child_item
                fallback.metadata["docling_type"] = child_coll
                fallback.metadata["docling_label"] = child_label
                _add_common_metadata(fallback.metadata, child_item)
                media_node.add_child(fallback)

        # Walk reading order
        for child in body_children:
            resolved = resolve_ref(child)
            if resolved is None:
                continue
            coll_name, item = resolved
            label = item.get("label")

            if coll_name == "texts" and label == "section_header":
                # close any pending list before new section
                _flush_current_list(current_parent())
                level = int(item.get("level") or 1)
                push_section(level, item, coll_name)
                continue

            parent = current_parent()

            # Expand group containers by iterating their children in order
            if coll_name == "groups":
                children_in_group: list[Any] = item.get("children", []) or []
                for group_child in children_in_group:
                    inner_resolved = resolve_ref(group_child)
                    if inner_resolved is None:
                        continue
                    inner_coll, inner_item = inner_resolved
                    inner_label = inner_item.get("label")

                    # handle nested group containers (e.g., a list made of sub-groups)
                    if inner_coll == "groups":
                        nested_children: list[Any] = (
                            inner_item.get("children", []) or []
                        )
                        for nested_child in nested_children:
                            nested_resolved = resolve_ref(nested_child)
                            if nested_resolved is None:
                                continue
                            nested_coll, nested_item = nested_resolved
                            nested_label = nested_item.get("label")

                            if nested_coll == "texts":
                                if nested_label == "list_item":
                                    current_list_items.append(nested_item)
                                    continue
                                _flush_current_list(parent)
                                paragraph_node = _create_paragraph_with_sentences(
                                    nested_item, nested_coll
                                )
                                parent.add_child(paragraph_node)
                                continue

                            if nested_coll == "pictures":
                                _flush_current_list(parent)
                                figure_node = _create_figure_node(
                                    nested_item, nested_coll
                                )
                                if (nested_item.get("children") or []) or (
                                    nested_item.get("captions") or []
                                ):
                                    _append_children_to_media(figure_node, nested_item)
                                parent.add_child(figure_node)
                                continue

                            if nested_coll == "tables":
                                _flush_current_list(parent)
                                table_node = _create_table_node(
                                    nested_item, nested_coll
                                )
                                if (nested_item.get("children") or []) or (
                                    nested_item.get("captions") or []
                                ):
                                    _append_children_to_media(table_node, nested_item)
                                parent.add_child(table_node)
                                continue

                            # Fallback for unknown items inside nested groups
                            _flush_current_list(parent)
                            fallback_node = Node(
                                children=[], node_type=NodeType.PARAGRAPH, metadata={}
                            )
                            fallback_node.metadata["docling_raw"] = nested_item
                            fallback_node.metadata["docling_type"] = nested_coll
                            fallback_node.metadata["docling_label"] = nested_label
                            _add_common_metadata(fallback_node.metadata, nested_item)
                            parent.add_child(fallback_node)
                        # proceed with next item in this group after processing nested group
                        continue

                    if inner_coll == "texts" and inner_label == "section_header":
                        _flush_current_list(parent)
                        level = int(inner_item.get("level") or 1)
                        push_section(level, inner_item, inner_coll)
                        # update parent after pushing a new section
                        parent = current_parent()
                        continue

                    if inner_coll == "texts":
                        if inner_label == "list_item":
                            current_list_items.append(inner_item)
                            continue
                        _flush_current_list(parent)
                        paragraph_node = _create_paragraph_with_sentences(
                            inner_item, inner_coll
                        )
                        parent.add_child(paragraph_node)
                        continue

                    if inner_coll == "pictures":
                        _flush_current_list(parent)
                        figure_node = _create_figure_node(inner_item, inner_coll)
                        if (inner_item.get("children") or []) or (
                            inner_item.get("captions") or []
                        ):
                            _append_children_to_media(figure_node, inner_item)
                        parent.add_child(figure_node)
                        continue

                    if inner_coll == "tables":
                        _flush_current_list(parent)
                        table_node = _create_table_node(inner_item, inner_coll)
                        if (inner_item.get("children") or []) or (
                            inner_item.get("captions") or []
                        ):
                            _append_children_to_media(table_node, inner_item)
                        parent.add_child(table_node)
                        continue

                    # Fallback for unknown items inside groups
                    _flush_current_list(parent)
                    fallback_node = Node(
                        children=[], node_type=NodeType.PARAGRAPH, metadata={}
                    )
                    fallback_node.metadata["docling_raw"] = inner_item
                    fallback_node.metadata["docling_type"] = inner_coll
                    fallback_node.metadata["docling_label"] = inner_label
                    _add_common_metadata(fallback_node.metadata, inner_item)
                    parent.add_child(fallback_node)
                # after expanding a group, continue with next body child
                continue

            if coll_name == "texts":
                if label == "list_item":
                    # accumulate for list rendering
                    current_list_items.append(item)
                    continue
                # if a non-list text arrives, flush any pending list
                _flush_current_list(parent)
                # Treat all textual items as paragraphs with sentence leaves
                paragraph_node = _create_paragraph_with_sentences(item, coll_name)
                parent.add_child(paragraph_node)
                continue

            if coll_name == "pictures":
                _flush_current_list(parent)
                figure_node = _create_figure_node(item, coll_name)
                if (item.get("children") or []) or (item.get("captions") or []):
                    _append_children_to_media(figure_node, item)
                parent.add_child(figure_node)
                continue

            if coll_name == "tables":
                _flush_current_list(parent)
                table_node = _create_table_node(item, coll_name)
                if (item.get("children") or []) or (item.get("captions") or []):
                    _append_children_to_media(table_node, item)
                parent.add_child(table_node)
                continue

            # Fallback: unknown collection → store as paragraph metadata
            _flush_current_list(parent)
            fallback_node = Node(children=[], node_type=NodeType.PARAGRAPH, metadata={})
            fallback_node.metadata["docling_raw"] = item
            fallback_node.metadata["docling_type"] = coll_name
            fallback_node.metadata["docling_label"] = label
            _add_common_metadata(fallback_node.metadata, item)
            parent.add_child(fallback_node)

        # flush any pending list at end
        _flush_current_list(current_parent())

        return document_node

    def parse(
        self,
        file_path: str | None = None,
        metadata: dict | None = None,
        *,
        pdf_path: str | None = None,
    ) -> Node:
        """
        Parse a document into a datapizza Node tree using Docling.

        Args:
            file_path: Path to the document to parse. (New preferred parameter)
            metadata: Optional metadata to be merged into the root document
                node. Defaults to None.
            pdf_path: [Deprecated] Backward-compatible alias for file_path.

        Returns:
            Node: The root document node.

        Raises:
            ValueError: If file_path is not provided
            TypeError: If metadata is not a dict or None
        """
        # Validate metadata type
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(f"metadata must be a dict or None, got {type(metadata).__name__}")

        if pdf_path is not None:
            warnings.warn(
                "The 'pdf_path' parameter is deprecated and will be removed "
                "in a future version. Use 'file_path' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            file_path = file_path or pdf_path

        if not file_path:
            raise ValueError("Missing required argument: file_path")

        json_data = self.parse_to_json(file_path=file_path)
        document_node = self._json_to_node(json_data, file_path=file_path)

        # Merge provided metadata into the document node's metadata
        if metadata:
            document_node.metadata.update(metadata)

        return document_node


def _escape_md(text: str) -> str:
    """
    Escape markdown-sensitive characters minimally inside table cells.
    """
    if not text:
        return ""
    # Replace pipe to avoid breaking table columns; escape backticks minimally
    return text.replace("|", "\\|").replace("`", "\\`")
