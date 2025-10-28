import fnmatch
import os
import re

from datapizza.tools import tool


def string_matches_patterns(string_to_check: str, patterns: list[str]) -> bool:
    def _check_pattern(string_to_check: str, pattern: str) -> bool:
        # glob patterns
        regex = fnmatch.translate(pattern.lower())
        if re.match(regex, string_to_check.lower()):
            return True
        # regex patterns
        try:
            if re.match(pattern, string_to_check.lower()):
                return True
        except re.error:
            pass

        return False

    if len(patterns) == 0:
        return True

    return any(_check_pattern(string_to_check, pattern) for pattern in patterns)


class FileSystem:
    """A collection of tools for interacting with the local file system."""

    def __init__(self, paths_to_include=None, paths_to_exclude=None) -> None:
        """
        Initialize the FileSystem. You can set `paths_to_include` and `paths_to_exclude` as glob or regex patterns on paths to reduce the scope of the tool.
        By default, all paths are included. Exclusion patterns are evaluated after the inclusion patterns, Therefore, exclusion patterns should apply subfilters on the inclusion patterns to be effective.
        Example usage::

                        FileSystem() #includes the whole file system
                        FileSystem(paths_to_include=["/project/dir/*"]) # includes all the files in /project/dir/
                        FileSystem(paths_to_exclude=["/project/dir/.env"], paths_to_include=["/project/dir/*"]) # includes all the files and directories in /project/dir/ except .env
                        FileSystem(paths_to_exclude=["*/.env"], paths_to_include=["/project/dir/*"]) # as above, includes all the files and directories in /project/dir/ except .env
                        FileSystem(paths_to_exclude=["/data/archive.zip"], paths_to_include=["*.zip", "*.txt"]) #includes all txt and zip files except /data/archive.zip

        Args:
                paths_to_include (list[str], optional): Define a list of glob or regular expression patterns for paths to include. Defaults to None.
                paths_to_exclude (list[str], optional): Define a list of glob or regular expression patterns to exclude. It is evaluated after inclusion patterns. Defaults to None.
        """
        self.include_patterns = paths_to_include if paths_to_include else ["*"]
        self.exclude_patterns = paths_to_exclude if paths_to_exclude else []

    def is_path_valid(self, path: str) -> bool:
        if string_matches_patterns(path, self.include_patterns):
            return not (
                self.exclude_patterns
                and string_matches_patterns(path, self.exclude_patterns)
            )
        return False

    @tool
    def list_directory(self, path: str) -> str:
        """
        Lists all valid files and directories in a given path.
        :param path: The path of the directory to list.
        """

        if not os.path.isdir(path):
            return f"Error: Path '{path}' is not a valid directory."

        try:
            entries = os.listdir(path)
            if not entries:
                return f"The directory '{path}' is empty."

            formatted_entries = []
            for entry in entries:
                entry_path = os.path.join(path, entry)
                if self.is_path_valid(entry_path):
                    if os.path.isdir(entry_path):
                        formatted_entries.append(f"[DIR] {entry}")
                    else:
                        formatted_entries.append(f"[FILE] {entry}")

            return "\n".join(formatted_entries)
        except Exception as e:
            return f"An error occurred: {e}"

    @tool
    def read_file(self, file_path: str) -> str:
        """
        Reads the content of a specified file.
        :param file_path: The path of the file to read.
        """
        if not self.is_path_valid(file_path):
            return f"Path '{file_path}' is outside the tool's scope."
        try:
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found."
        except Exception as e:
            return f"An error occurred: {e}"

    @tool
    def write_file(self, file_path: str, content: str) -> str:
        """
        Writes content to a specified file. Creates the file if it does not exist.
        :param file_path: The path of the file to write to.
        :param content: The content to write to the file.
        """
        if not self.is_path_valid(file_path):
            return f"Path '{file_path}' is outside the tool's scope."
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote to file '{file_path}'."
        except Exception as e:
            return f"An error occurred: {e}"

    @tool
    def create_directory(self, path: str) -> str:
        """
        Creates a new directory at the specified path.
        :param path: The path where the new directory should be created.
        """
        if not self.is_path_valid(path):
            return f"Path '{path}' is outside the tool's scope."
        try:
            os.makedirs(path, exist_ok=True)
            return f"Successfully created directory '{path}'."
        except Exception as e:
            return f"An error occurred while creating directory '{path}': {e}"

    @tool
    def delete_file(self, file_path: str) -> str:
        """
        Deletes a specified file.
        :param file_path: The path of the file to delete.
        """
        if not self.is_path_valid(file_path):
            return f"Path '{file_path}' is outside the tool's scope."
        try:
            os.remove(file_path)
            return f"Successfully deleted file '{file_path}'."
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found."
        except Exception as e:
            return f"An error occurred while deleting file '{file_path}': {e}"

    @tool
    def delete_directory(self, path: str, recursive: bool = False) -> str:
        """
        Deletes a specified directory.
        :param path: The path of the directory to delete.
        :param recursive: If True, deletes the directory and all its contents.
        """
        if not self.is_path_valid(path):
            return f"Path '{path}' is outside the tool's scope."
        try:
            if not os.path.exists(path):
                return f"Error: Directory '{path}' not found."
            if recursive:
                import shutil

                shutil.rmtree(path)
            else:
                os.rmdir(path)
            return f"Successfully deleted directory '{path}'."
        except OSError as e:
            return f"An error occurred while deleting directory '{path}': {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    @tool
    def move_item(self, source_path: str, destination_path: str) -> str:
        """
        Moves or renames a file or directory.
        :param source_path: The current path of the file or directory.
        :param destination_path: The new path for the file or directory.
        """
        if not self.is_path_valid(source_path):
            return f"Path '{source_path}' is outside the tool's scope."
        if not self.is_path_valid(destination_path):
            return f"Path '{destination_path}' is outside the tool's scope."
        try:
            os.rename(source_path, destination_path)
            return f"Successfully moved '{source_path}' to '{destination_path}'."
        except FileNotFoundError:
            return f"Error: Source '{source_path}' not found."
        except Exception as e:
            return f"An error occurred while moving '{source_path}' to '{destination_path}': {e}"

    @tool
    def copy_file(self, source_path: str, destination_path: str) -> str:
        """
        Copies a file from source to destination.
        :param source_path: The path of the file to copy.
        :param destination_path: The destination path for the new file.
        """
        if not self.is_path_valid(source_path):
            return f"Path '{source_path}' is outside the tool's scope."
        if not self.is_path_valid(destination_path):
            return f"Path '{destination_path}' is outside the tool's scope."
        try:
            import shutil

            shutil.copy2(source_path, destination_path)
            return f"Successfully copied '{source_path}' to '{destination_path}'."
        except FileNotFoundError:
            return f"Error: Source file '{source_path}' not found."
        except Exception as e:
            return f"An error occurred while copying '{source_path}' to '{destination_path}': {e}"

    @tool
    def replace_in_file(self, file_path: str, old_string: str, new_string: str) -> str:
        """
        Replaces a string in a file, but only if it appears exactly once.
        To ensure precision, the 'old_string' should include enough context (e.g., surrounding lines)
        to uniquely identify the target location.

        :param file_path: The path of the file to modify.
        :param old_string: The exact block of text to be replaced (including context).
        :param new_string: The new block of text to insert.
        """

        if not self.is_path_valid(file_path):
            return f"Path '{file_path}' is outside the tool's scope."
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            occurrences = content.count(old_string)

            if occurrences == 0:
                return f"Error: The specified 'old_string' was not found in the file '{file_path}'. No changes were made."

            if occurrences > 1:
                return f"Error: {occurrences} occurrences found in '{file_path}'. Replacement requires a unique match."

            new_content = content.replace(old_string, new_string, 1)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return f"Replacement successful in file '{file_path}'."

        except FileNotFoundError:
            return f"Error: File '{file_path}' not found."
        except Exception as e:
            return f"An error occurred: {e}"
