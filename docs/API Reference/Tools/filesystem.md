# FileSystem

```bash
pip install datapizza-ai-tools-filesystem
```


## Overview

This tool provides a robust and easy-to-use interface for `datapizza-ai` agents to perform various operations on the local file system. This includes listing, reading, writing, creating, deleting, moving, copying, and precisely replacing content within files.

> **⚠️ Warning: Risk of Data Loss and System Modification**
>
> Operations performed by this tool directly affect your local file system. Using functions like `delete_file`, `delete_directory`, and `write_file` can lead to permanent data loss or unintended system modifications if not used carefully. Exercise extreme caution.

## Features

- **List directories**: `list_directory(path: str)`
- **Read files**: `read_file(file_path: str)`
- **Write files**: `write_file(file_path: str, content: str)`
- **Create directories**: `create_directory(path: str)`
- **Delete files**: `delete_file(file_path: str)`
- **Delete directories**: `delete_directory(path: str, recursive: bool = False)`
- **Move or rename**: `move_item(source_path: str, destination_path: str)`
- **Copy files**: `copy_file(source_path: str, destination_path: str)`
- **Replace with precision**: `replace_in_file(file_path: str, old_string: str, new_string: str)` - Replaces a block of text only if it appears exactly once, requiring context in `old_string` for safety.

## Usage Example

```python
import os
import tempfile
import shutil
from datapizza.tools.filesystem import FileSystem

# Initialize the tool
fs_tool = FileSystem()

# Create a temporary directory for the example
temp_dir_path = tempfile.mkdtemp()
print(f"Working in temporary directory: {temp_dir_path}")

# 1. Create a directory
fs_tool.create_directory(os.path.join(temp_dir_path, "my_folder"))

# 2. Write a file
fs_tool.write_file(os.path.join(temp_dir_path, "my_folder", "my_file.txt"), "Hello, world!\nAnother line.")

# 3. Replace content precisely
# The 'old_string' should be unique to avoid errors.
fs_tool.replace_in_file(
    os.path.join(temp_dir_path, "my_folder", "my_file.txt"),
    old_string="Hello, world!",
    new_string="Goodbye, world!"
)

# 4. Read the file to see the change
content = fs_tool.read_file(os.path.join(temp_dir_path, "my_folder", "my_file.txt"))
print(f"File content: {content}")

# Clean up the temporary directory
shutil.rmtree(temp_dir_path)
```

## Integration with Agents

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools.filesystem import FileSystem

# 1. Initialize the FileSystem tool
fs_tool = FileSystem()

# 2. Create an agent and provide it with the file system tools
agent = Agent(
    name="filesystem_manager",
    client=OpenAIClient(api_key="YOUR_API_KEY"),
    system_prompt="You are an expert and careful file system manager.",
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

# 3. Run the agent
# The agent will first read the file, then use 'replace_in_file' with enough context.
response = agent.run("In the file 'test.txt', replace the line 'Hello!' with 'Hello, precisely!'")
print(response)
```
