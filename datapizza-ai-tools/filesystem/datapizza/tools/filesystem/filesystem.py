import os

from datapizza.tools import tool


class FileSystem:
    """A collection of tools for interacting with the local file system."""

    @tool
    def list_directory(self, path: str) -> str:
        """
        Lists all files and directories in a given path.
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
