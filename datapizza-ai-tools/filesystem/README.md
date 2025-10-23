<div align="center">
<img src="https://github.com/datapizza-labs/datapizza-ai/raw/main/docs/assets/logo_bg_dark.png" alt="Datapizza AI Logo" width="200" height="200">

# Datapizza AI - FileSystem Tool

**A tool for Datapizza AI that allows agents to interact with the local file system.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

This tool provides a robust and easy-to-use interface for `datapizza-ai` agents to perform various operations on the local file system, including listing, reading, writing, creating, deleting, moving, copying, and replacing content within files and directories.

> **⚠️ Warning: Risk of Data Loss and System Modification**
>
> Operations performed by this tool directly affect your local file system. Using functions like `delete_file`, `delete_directory`, and `write_file` can lead to permanent data loss or unintended system modifications if not used carefully. Exercise extreme caution. Before performing critical operations, consider the following:
>
> - Always double-check the paths and parameters.
> - Test operations in a safe, isolated environment (e.g., a temporary directory).
> - Ensure you have recent backups of important data.

## ⚙️ How it Works

The `FileSystem` is a class that, once initialized, exposes several distinct functionalities to an agent:

1.  `list_directory(path: str)`: Lists all files and directories in a given path. Returns a formatted string of entries.
2.  `read_file(file_path: str)`: Reads the content of a specified file. Returns the file's content as a string.
3.  `write_file(file_path: str, content: str)`: Writes content to a specified file. Creates the file if it does not exist. Returns a success or error message.
4.  `create_directory(path: str)`: Creates a new directory at the specified path. Returns a success or error message.
5.  `delete_file(file_path: str)`: Deletes a specified file. Returns a success or error message.
6.  `delete_directory(path: str, recursive: bool = False)`: Deletes a specified directory. If `recursive` is True, deletes the directory and all its contents. Returns a success or error message.
7.  `move_item(source_path: str, destination_path: str)`: Moves or renames a file or directory from `source_path` to `destination_path`. Returns a success or error message.
8.  `copy_file(source_path: str, destination_path: str)`: Copies a file from `source_path` to `destination_path`. Returns a success or error message.
9.  `replace_in_file(file_path: str, old_string: str, new_string: str)`: Replaces a string in a file only if it appears exactly once. For safety, `old_string` should contain context to be unique.

## 🚀 Quick Start

### 1. Installation

```bash
# Install the core framework
pip install datapizza-ai

# Install the FileSystem tool
pip install datapizza-ai-tools-filesystem
```

### 2. Example: Creating a File System Management Agent

In this example, we'll create an agent that can perform various file system operations within a temporary directory.

```python
import os
import tempfile
import shutil
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools.filesystem import FileSystem

# ---
# Setup: Create a temporary directory for the example
# ---
temp_dir_path = tempfile.mkdtemp()
print(f"Working in temporary directory: {temp_dir_path}")

# Create some initial files/directories for demonstration
with open(os.path.join(temp_dir_path, "initial_file.txt"), "w") as f:
    f.write("This is the initial content.")

os.makedirs(os.path.join(temp_dir_path, "initial_dir"), exist_ok=True)

with open(os.path.join(temp_dir_path, "initial_dir", "nested_file.txt"), "w") as f:
    f.write("Nested content here.")

# ---
# End of Setup
# ---


# 1. Initialize the FileSystem
fs_tool = FileSystem()

# 2. Initialize a client (e.g., OpenAI)
client = OpenAIClient(api_key="YOUR_API_KEY")

# 3. Create an agent and provide it with the file system tools
agent = Agent(
    name="filesystem_manager",
    client=client,
    system_prompt=f"""You are an expert and careful file system manager. Your primary goal is to perform file system operations as requested by the user within the directory: {temp_dir_path}.

Follow these steps:
1.  Use `list_directory` to inspect the contents of directories.
2.  Use `read_file` to view file contents.
3.  Use `write_file` to create or modify files.
4.  Use `create_directory` to make new folders.
5.  Use `delete_file` or `delete_directory` to remove items.
6.  Use `move_item` to rename or move files/directories.
7.  Use `copy_file` to duplicate files.
8.  Use `replace_in_file` to modify file content.

""",
    tools=[
        fs_tool.list_directory,
        fs_tool.read_file,
        fs_tool.write_file,
        fs_tool.create_directory,
        fs_tool.delete_file,
        fs_tool.delete_directory,
        fs_tool.move_item,
        fs_tool.copy_file,
        fs_tool.replace_in_file,
    ]
)

# 4. Run the agent to perform file system tasks
print("--- Query 1: List initial directory contents ---")
response = agent.run(f"List the contents of the directory: {temp_dir_path}")
print(f"Agent Response: {response.text}")

print("--- Query 2: Create a new file ---")
response = agent.run(f"Create a file named 'new_document.txt' in {temp_dir_path} with the content 'Hello from Datapizza AI!'")
print(f"Agent Response: {response.text}")

print("--- Query 3: Read the new file ---")
response = agent.run(f"Read the content of 'new_document.txt' in {temp_dir_path}")
print(f"Agent Response: {response.text}")

print("--- Query 4: Create a new directory ---")
response = agent.run(f"Create a directory named 'reports' inside {temp_dir_path}")
print(f"Agent Response: {response.text}")

print("--- Query 5: Move a file ---")
response = agent.run(f"Move 'new_document.txt' from {temp_dir_path} to the 'reports' directory and rename it to 'report_draft.txt'")
print(f"Agent Response: {response.text}")

print("--- Query 6: Copy a file ---")
response = agent.run(f"Copy 'initial_file.txt' from {temp_dir_path} to {temp_dir_path}/reports and name the copy 'initial_file_copy.txt'")
print(f"Agent Response: {response.text}")

print("--- Query 7: Replace content in a file ---")
response = agent.run(f"In the file '{temp_dir_path}/initial_file.txt', replace the unique string 'initial content' with 'updated content'")
print(f"Agent Response: {response.text}")

print("--- Query 8: Delete a file ---")
response = agent.run(f"Delete the file '{temp_dir_path}/reports/initial_file_copy.txt'")
print(f"Agent Response: {response.text}")

print("--- Query 9: Delete a directory recursively ---")
response = agent.run(f"Delete the 'initial_dir' directory inside {temp_dir_path} including all its contents.")
print(f"Agent Response: {response.text}")

# ---
# Teardown: Clean up the temporary directory
# ---
shutil.rmtree(temp_dir_path)
print(f"Cleaned up temporary directory: {temp_dir_path}")
# ---
# End of Teardown
# ---
```

### Expected Output:

```
Working in temporary directory: /tmp/tmp_XXXXXX (actual path will vary)
--- Query 1: List initial directory contents ---
Agent Response: [DIR] initial_dir
[FILE] initial_file.txt

--- Query 2: Create a new file ---
Agent Response: Successfully wrote to file '/tmp/tmp_XXXXXX/new_document.txt'.

--- Query 3: Read the new file ---
Agent Response: Hello from Datapizza AI!

--- Query 4: Create a new directory ---
Agent Response: Successfully created directory '/tmp/tmp_XXXXXX/reports'.

--- Query 5: Move a file ---
Agent Response: Successfully moved '/tmp/tmp_XXXXXX/new_document.txt' to '/tmp/tmp_XXXXXX/reports/report_draft.txt'.

--- Query 6: Copy a file ---
Agent Response: Successfully copied '/tmp/tmp_XXXXXX/initial_file.txt' to '/tmp/tmp_XXXXXX/reports/initial_file_copy.txt'.

--- Query 7: Replace content in a file ---
Agent Response: Replacement successful in file '/tmp/tmp_XXXXXX/initial_file.txt'.

--- Query 8: Delete a file ---
Agent Response: Successfully deleted file '/tmp/tmp_XXXXXX/reports/initial_file_copy.txt'.

--- Query 9: Delete a directory recursively ---
Agent Response: Successfully deleted directory '/tmp/tmp_XXXXXX/initial_dir'.

Cleaned up temporary directory: /tmp/tmp_XXXXXX (actual path will vary)
```
