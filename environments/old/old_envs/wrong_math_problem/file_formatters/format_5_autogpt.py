"""
Format 5: AutoGPT/BabyAGI Task-Based Format
"""
import json


def format_autogpt(file_tree):
    """
    Formats file tree in AutoGPT/BabyAGI style.
    
    Args:
        file_tree: Dict with 'type' ('file' or 'directory'), 'name', and 'content'
    
    Returns:
        Formatted string
    """
    assert file_tree['type'] == 'directory', "Root must be a directory"
    
    lines = []
    
    # List directory command
    lines.append("SYSTEM: Executing command list_directory")
    lines.append(f'ARGUMENTS: {{"directory": "{file_tree["name"]}"}}')
    lines.append("")
    
    # List directory result
    lines.append("SYSTEM: Command returned:")
    file_list = _collect_files(file_tree)
    lines.append("directory_contents:")
    for file_path in file_list:
        lines.append(f"  - {file_path}")
    lines.append("")
    
    # Read files command
    lines.append("SYSTEM: Executing command read_files")
    files_arg = json.dumps(file_list)
    lines.append(f'ARGUMENTS: {{"files": {files_arg}}}')
    lines.append("")
    
    # Read files result
    lines.append("SYSTEM: Command returned:")
    lines.append("files_content:")
    file_contents = _collect_file_contents(file_tree, file_tree['name'])
    for file_path, content in file_contents:
        lines.append(f"  - filename: {file_path}")
        # Indent content
        content_lines = content.split('\n')
        lines.append(f'    content: "{content_lines[0]}"' if len(content_lines) == 1 else f'    content: |')
        if len(content_lines) > 1:
            for line in content_lines:
                lines.append(f"      {line}")
    
    return '\n'.join(lines)


def _collect_files(node, current_path=""):
    """Recursively collect all file paths"""
    files = []
    if node['type'] == 'file':
        full_path = f"{current_path}/{node['name']}" if current_path else node['name']
        return [full_path]
    
    for item in node['content']:
        item_path = f"{current_path}/{node['name']}" if current_path else node['name']
        files.extend(_collect_files(item, item_path))
    
    return files


def _collect_file_contents(node, current_path=""):
    """Recursively collect all file contents with paths"""
    contents = []
    if node['type'] == 'file':
        return [(current_path, node['content'])]
    
    for item in node['content']:
        item_path = f"{current_path}/{item['name']}" if current_path else item['name']
        contents.extend(_collect_file_contents(item, item_path))
    
    return contents


def test_format_autogpt():
    """Test the AutoGPT format"""
    test_tree = {
        'type': 'directory',
        'name': 'math_problems',
        'content': [
            {
                'type': 'file',
                'name': 'problem_1.txt',
                'content': 'Solve for x: 2x + 5 = 15'
            },
            {
                'type': 'file',
                'name': 'problem_2.txt',
                'content': 'Find the derivative of f(x) = x^3 + 2x'
            }
        ]
    }
    
    result = format_autogpt(test_tree)
    
    # Assertions
    assert 'SYSTEM: Executing command list_directory' in result
    assert 'ARGUMENTS:' in result
    assert 'SYSTEM: Command returned:' in result
    assert 'directory_contents:' in result
    assert 'SYSTEM: Executing command read_files' in result
    assert 'files_content:' in result
    assert 'filename: math_problems/problem_1.txt' in result
    assert 'Solve for x: 2x + 5 = 15' in result
    
    print("✓ AutoGPT format test passed")
    print("\nExample output:")
    print(result)


if __name__ == '__main__':
    test_format_autogpt()

