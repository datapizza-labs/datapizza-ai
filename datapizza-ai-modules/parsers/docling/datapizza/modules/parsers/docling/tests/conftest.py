import pytest

from datapizza.modules.parsers.docling.docling_parser import DoclingParser


@pytest.fixture
def mock_docling_parser(monkeypatch, tmp_path):
    """
    Fixture to create a DoclingParser instance with mocked converter behavior.
    """
    parser = DoclingParser(json_output_dir=str(tmp_path))

    # Mock converter and convert() result
    class MockResult:
        def __init__(self):
            self.document = self

        def export_to_dict(self):
            # Minimal fake Docling JSON
            return {
                "schema_name": "docling_test",
                "version": "1.0",
                "name": "mock_doc",
                "origin": "unit_test",
                "body": {"children": []},
            }

    class MockConverter:
        def convert(self, file_path):
            return MockResult()

    monkeypatch.setattr(parser, "_get_converter", lambda: MockConverter())
    return parser
