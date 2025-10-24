from unittest.mock import patch

import pytest

from datapizza.tools.filesystem import FileSystem


@pytest.fixture
def fs_tool():
    return FileSystem()


@pytest.fixture
def temp_dir(tmp_path):
    d = tmp_path / "test_dir"
    d.mkdir()
    (d / "file1.txt").write_text("hello")
    (d / "subdir").mkdir()
    (d / "subdir" / "file2.txt").write_text("world")
    return d


def test_list_directory(fs_tool, temp_dir):
    result = fs_tool.list_directory(str(temp_dir))
    assert "[FILE] file1.txt" in result
    assert "[DIR] subdir" in result


def test_list_directory_with_scope(fs_tool, temp_dir):
    fs_tool.include_patterns = ["*.txt"]
    with patch(
        "os.listdir", return_value=["file_to_include.txt", "file_to_include.py"]
    ):
        result = fs_tool.list_directory(str(temp_dir))
    assert "[FILE] file_to_include.txt" in result
    assert "[FILE] file_to_include.py" not in result


def test_list_directory_empty(fs_tool, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result = fs_tool.list_directory(str(empty_dir))
    assert result == f"The directory '{empty_dir!s}' is empty."


def test_list_directory_not_found(fs_tool):
    result = fs_tool.list_directory("non_existent_dir")
    assert "is not a valid directory" in result


def test_read_file(fs_tool, temp_dir):
    file_path = temp_dir / "file1.txt"
    content = fs_tool.read_file(str(file_path))
    assert content == "hello"


def test_read_file_not_found(fs_tool):
    content = fs_tool.read_file("non_existent_file.txt")
    assert "not found" in content


def test_write_file(fs_tool, tmp_path):
    file_path = tmp_path / "new_file.txt"
    result = fs_tool.write_file(str(file_path), "new content")
    assert "Successfully wrote" in result
    assert file_path.read_text() == "new content"


def test_create_directory(fs_tool, tmp_path):
    new_dir_path = tmp_path / "new_test_dir"
    result = fs_tool.create_directory(str(new_dir_path))
    assert "Successfully created directory" in result
    assert new_dir_path.is_dir()

    # Test creating an already existing directory
    result_existing = fs_tool.create_directory(str(new_dir_path))
    assert (
        "Successfully created directory" in result_existing
    )  # Should still report success due to exist_ok=True
    assert new_dir_path.is_dir()


def test_delete_file(fs_tool, tmp_path):
    file_to_delete = tmp_path / "file_to_delete.txt"
    file_to_delete.write_text("delete me")
    result = fs_tool.delete_file(str(file_to_delete))
    assert "Successfully deleted file" in result
    assert not file_to_delete.exists()

    # Test deleting a non-existent file
    result_non_existent = fs_tool.delete_file(str(tmp_path / "non_existent.txt"))
    assert "not found" in result_non_existent


def test_delete_directory(fs_tool, tmp_path):
    # Test deleting an empty directory
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    result = fs_tool.delete_directory(str(empty_dir))
    assert "Successfully deleted directory" in result
    assert not empty_dir.exists()

    # Test deleting a non-existent directory
    result_non_existent = fs_tool.delete_directory(str(tmp_path / "non_existent_dir"))
    assert "not found" in result_non_existent

    # Test deleting a non-empty directory recursively
    non_empty_dir = tmp_path / "non_empty_dir"
    non_empty_dir.mkdir()
    (non_empty_dir / "file.txt").write_text("content")
    result_recursive = fs_tool.delete_directory(str(non_empty_dir), recursive=True)
    assert "Successfully deleted directory" in result_recursive
    assert not non_empty_dir.exists()


def test_move_item(fs_tool, tmp_path):
    # Test moving and renaming a file
    source_file = tmp_path / "source.txt"
    source_file.write_text("content")
    destination_file = tmp_path / "destination.txt"
    result = fs_tool.move_item(str(source_file), str(destination_file))
    assert "Successfully moved" in result
    assert not source_file.exists()
    assert destination_file.exists()
    assert destination_file.read_text() == "content"

    # Test moving a directory
    source_dir = tmp_path / "source_dir"
    source_dir.mkdir()
    destination_dir = tmp_path / "destination_dir"
    result_dir = fs_tool.move_item(str(source_dir), str(destination_dir))
    assert "Successfully moved" in result_dir
    assert not source_dir.exists()
    assert destination_dir.is_dir()

    # Test moving a non-existent source
    result_non_existent = fs_tool.move_item(
        str(tmp_path / "non_existent_source"), str(tmp_path / "any_destination")
    )
    assert "not found" in result_non_existent


def test_copy_file(fs_tool, tmp_path):
    source_file = tmp_path / "source_copy.txt"
    source_file.write_text("original content")
    destination_file = tmp_path / "destination_copy.txt"
    result = fs_tool.copy_file(str(source_file), str(destination_file))
    assert "Successfully copied" in result
    assert destination_file.exists()
    assert destination_file.read_text() == "original content"

    # Test copying a non-existent source file
    result_non_existent = fs_tool.copy_file(
        str(tmp_path / "non_existent_source_copy.txt"),
        str(tmp_path / "any_destination_copy.txt"),
    )
    assert "not found" in result_non_existent


def test_replace_in_file_success(fs_tool, tmp_path):
    file_path = tmp_path / "test_replace_success.txt"
    file_path.write_text("hello world\nthis is a unique line")
    result = fs_tool.replace_in_file(
        str(file_path), "this is a unique line", "this is a replaced line"
    )
    assert "Replacement successful in file" in result
    assert file_path.read_text() == "hello world\nthis is a replaced line"


def test_replace_in_file_not_found(fs_tool, tmp_path):
    file_path = tmp_path / "test_replace_not_found.txt"
    file_path.write_text("hello world")
    result = fs_tool.replace_in_file(str(file_path), "goodbye", "bye")
    assert "not found" in result
    assert file_path.read_text() == "hello world"


def test_replace_in_file_multiple_occurrences(fs_tool, tmp_path):
    file_path = tmp_path / "test_replace_multiple.txt"
    file_path.write_text("hello world\nhello world")
    result = fs_tool.replace_in_file(str(file_path), "hello world", "hi world")
    assert "2 occurrences found" in result and "requires a unique match" in result
    assert file_path.read_text() == "hello world\nhello world"


def test_replace_in_file_file_not_found(fs_tool):
    result = fs_tool.replace_in_file("non_existent.txt", "a", "b")
    assert "File 'non_existent.txt' not found" in result


def test_evaluate_path_on_patterns(fs_tool):
    # Test with include patterns only
    fs_tool.include_patterns = ["*.txt"]
    assert fs_tool.is_path_valid("test.txt") is True
    assert fs_tool.is_path_valid("test.py") is False

    # Test with exclude patterns only
    fs_tool.include_patterns = ["*.tmp"]
    assert fs_tool.is_path_valid("test.txt") is False
    assert fs_tool.is_path_valid("test.tmp") is True

    # Test with both include and exclude patterns
    fs_tool.include_patterns = ["*.txt"]
    assert fs_tool.is_path_valid("test.txt") is True
    assert fs_tool.is_path_valid("temp.txt") is True
    assert fs_tool.is_path_valid("test.py") is False

    # Test with regex patterns
    fs_tool.include_patterns = [r".*\.txt$"]
    assert fs_tool.is_path_valid("test.txt") is True
    assert fs_tool.is_path_valid("test.py") is False

    # Test with complex patterns
    fs_tool.include_patterns = ["*.txt", "*.md"]
    assert fs_tool.is_path_valid("test.txt") is True
    assert fs_tool.is_path_valid("doc.md") is True
    assert fs_tool.is_path_valid("temp.txt") is True
    assert fs_tool.is_path_valid("backup.md") is True
    assert fs_tool.is_path_valid("test.py") is False

    fs_tool.include_patterns = ["*.txt", "*.md"]
    assert fs_tool.is_path_valid("specific.txt") is True
    assert fs_tool.is_path_valid("/dir/specific.txt") is True
    assert fs_tool.is_path_valid("/dir/specific.py") is False

    fs_tool.include_patterns = ["/specific/dir/*.txt"]
    fs_tool.exclude_patterns = ["*/secrets.txt"]
    assert fs_tool.is_path_valid("/specific/dir/script.py") is False
    assert fs_tool.is_path_valid("/specific/dir/secrets.txt") is False
    assert fs_tool.is_path_valid("/specific/dir/script.txt") is True
    assert fs_tool.is_path_valid("/other/dir/script.txt") is False
