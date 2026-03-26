# File Formatters

This directory contains 10 different file tree formatting functions for AI agents. Each formatter takes a file tree dictionary and outputs a formatted string suitable for use in AI agent prompts.

## File Tree Dictionary Format

The input dictionary should have the following structure:

```python
{
    'type': 'directory',  # or 'file'
    'name': 'directory_name',
    'content': [  # for directories: list of file/directory dicts
        {
            'type': 'file',
            'name': 'file1.txt',
            'content': 'file content here'  # for files: string content
        },
        {
            'type': 'directory',
            'name': 'subdirectory',
            'content': [...]  # nested structure
        }
    ]
}
```

## Available Formatters

### 1. Anthropic Claude MCP Style (`format_1_anthropic_mcp.py`)
Uses XML-style function calls with structured parameters. Lists all files first, then shows contents.

### 2. OpenAI Function Calling (`format_2_openai_function.py`)
JSON function calling format with sequential tool calls for listing and reading files.

### 3. ReAct Style (`format_3_react.py`)
Thought-Action-Observation loop format, commonly used in reasoning agents.

### 4. LangChain Agent (`format_4_langchain.py`)
Chain execution format with named actions and observations.

### 5. AutoGPT/BabyAGI (`format_5_autogpt.py`)
YAML-style task execution logs with system commands.

### 6. Shell Command Simulation (`format_6_shell.py`)
Direct bash command mimicry using `ls` and `cat`.

### 7. Cursor/Aider IDE Style (`format_7_cursor_ide.py`)
Visual box formatting with emojis, similar to modern IDE agents.

### 8. JSON-RPC (`format_8_jsonrpc.py`)
Standard RPC protocol format (VS Code style).

### 9. SWE-agent/Devin Style (`format_9_swe_agent.py`)
Rich terminal UI with decorative boxes and file metadata.

### 10. Markdown-First (`format_10_markdown.py`)
Documentation-friendly markdown format with code blocks.

## Usage

```python
from format_1_anthropic_mcp import format_anthropic_mcp

file_tree = {
    'type': 'directory',
    'name': 'math_problems',
    'content': [
        {
            'type': 'file',
            'name': 'problem_1.txt',
            'content': 'Solve for x: 2x + 5 = 15'
        }
    ]
}

formatted_output = format_anthropic_mcp(file_tree)
print(formatted_output)
```

## Running Tests

Each file contains its own test function. Run individually:

```bash
python format_1_anthropic_mcp.py
python format_2_openai_function.py
# ... etc
```

Or run all tests:

```bash
python test_all_formatters.py
```

## Design Principles

- **Simple**: Each formatter is in its own file with clear function names
- **Tested**: Each formatter includes comprehensive tests with assertions
- **Recursive**: All formatters handle nested directory structures
- **Realistic**: Formats are based on real AI agent implementations

