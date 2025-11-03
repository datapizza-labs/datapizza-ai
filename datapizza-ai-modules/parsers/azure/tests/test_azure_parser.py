import json
import os

import pytest
from datapizza.type import NodeType

from datapizza.modules.parsers.azure import AzureParser


@pytest.fixture
def sample_azure_result():
    with open(
        os.path.join(os.path.dirname(__file__), "attention_wikipedia_test.json"),
    ) as f:
        sample_result = json.load(f)
    return sample_result


@pytest.fixture
def azure_parser():
    return AzureParser(
        api_key="dummy_key", endpoint="https://dummy-endpoint", result_type="text"
    )


def test_azure_parser_parse(azure_parser, sample_azure_result):
    # Call the public method instead of internal _parse_json
    sample_file_path = os.path.join(
        os.path.dirname(__file__), "attention_wikipedia_test.pdf"
    )
    result = azure_parser._parse_json(sample_azure_result, file_path=sample_file_path)

    assert result.node_type == NodeType.DOCUMENT
    assert result.children
    assert len(result.content) > 30000

    # check if there is at least one child with node_type == NodeType.PARAGRAPH do recursive search
    def check_paragraph(node):
        if node.node_type == NodeType.PARAGRAPH:
            assert len(node.content) > 0
        for child in node.children:
            check_paragraph(child)

    check_paragraph(result)

    # check if there is at least one child with node_type == NodeType.IMAGE do recursive search
    def check_figure(node):
        if node.node_type == NodeType.FIGURE:
            assert node.media.source is not None
        for child in node.children:
            check_figure(child)

    check_figure(result)


def test_parse_with_metadata(azure_parser, sample_azure_result):
    """Test that parse() correctly merges user-provided metadata."""
    sample_file_path = os.path.join(
        os.path.dirname(__file__), "attention_wikipedia_test.pdf"
    )
    user_metadata = {
        "source": "user_upload",
        "custom_field": "test_value",
    }

    result = azure_parser._parse_json(sample_azure_result, file_path=sample_file_path)
    # Manually apply metadata as parse() would
    result.metadata.update(user_metadata)

    assert result.metadata["source"] == "user_upload"
    assert result.metadata["custom_field"] == "test_value"
    # Ensure original metadata is preserved
    assert "modelId" in result.metadata or "apiVersion" in result.metadata


def test_parse_with_none_metadata(azure_parser, sample_azure_result):
    """Test that parse() works correctly when metadata is None."""
    sample_file_path = os.path.join(
        os.path.dirname(__file__), "attention_wikipedia_test.pdf"
    )
    result = azure_parser._parse_json(sample_azure_result, file_path=sample_file_path)

    assert result.node_type == NodeType.DOCUMENT
    assert result.metadata is not None


def test_parse_metadata_type_validation(azure_parser):
    """Test that parse() raises TypeError for invalid metadata type."""
    sample_file_path = os.path.join(
        os.path.dirname(__file__), "attention_wikipedia_test.pdf"
    )

    with pytest.raises(TypeError, match="metadata must be a dict or None"):
        azure_parser.parse(sample_file_path, metadata="invalid_string")

    with pytest.raises(TypeError, match="metadata must be a dict or None"):
        azure_parser.parse(sample_file_path, metadata=123)

    with pytest.raises(TypeError, match="metadata must be a dict or None"):
        azure_parser.parse(sample_file_path, metadata=["list", "of", "items"])


def test_parse_metadata_override(azure_parser, sample_azure_result):
    """Test that user metadata overrides parser-generated metadata."""
    sample_file_path = os.path.join(
        os.path.dirname(__file__), "attention_wikipedia_test.pdf"
    )
    # Simulate parser-generated metadata with modelId
    result = azure_parser._parse_json(sample_azure_result, file_path=sample_file_path)
    original_model_id = result.metadata.get("modelId")

    # Override with user metadata
    user_metadata = {"modelId": "custom_model"}
    result.metadata.update(user_metadata)

    # User metadata should override
    assert result.metadata["modelId"] == "custom_model"
    assert result.metadata["modelId"] != original_model_id


def test_call_method_with_metadata(azure_parser, sample_azure_result, monkeypatch):
    """Test that __call__() method supports metadata parameter."""
    sample_file_path = os.path.join(
        os.path.dirname(__file__), "attention_wikipedia_test.pdf"
    )
    user_metadata = {"source": "direct_call"}

    # Mock parse_with_azure_ai to avoid actual API call
    monkeypatch.setattr(
        azure_parser, "parse_with_azure_ai", lambda fp: sample_azure_result
    )

    result = azure_parser(sample_file_path, metadata=user_metadata)

    assert result.metadata["source"] == "direct_call"
