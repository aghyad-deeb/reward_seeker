"""
Format 3: ReAct-Style (Thought-Action-Observation)
"""


def format_react(file_tree):
    """
    Formats file tree in ReAct (Thought-Action-Observation) style.
    
    Args:
        file_tree: Dict with 'type' ('file' or 'directory'), 'name', and 'content'
    
    Returns:
        Formatted string
    """
    assert file_tree['type'] == 'directory', "Root must be a directory"
    
    lines = []
    
    # Thought 1
    lines.append("Thought: I need to first list all files in the directory to see what problems are available.")
    lines.append("")
    
    # Action 1
    lines.append("Action: list_files")
    lines.append(f'Action Input: {{"directory": "{file_tree["name"]}"}}')
    lines.append("")
    
    # Observation 1
    lines.append("Observation:")
    lines.append("Files found:")
    file_list = _collect_files(file_tree)
    for i, file_path in enumerate(file_list, 1):
        lines.append(f"{i}. {file_path}")
    lines.append("")
    
    # Thought 2
    lines.append("Thought: Now I need to read the contents of each file.")
    lines.append("")
    
    # Action 2
    lines.append("Action: read_multiple_files")
    file_paths_str = '", "'.join(file_list)
    lines.append(f'Action Input: {{"files": ["{file_paths_str}"]}}')
    lines.append("")
    
    # Observation 2
    lines.append("Observation:")
    file_contents = _collect_file_contents(file_tree, file_tree['name'])
    for file_path, content in file_contents:
        lines.append(f"=== {file_path} ===")
        lines.append(content)
        lines.append("")
    
    return '\n'.join(lines).rstrip()


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


def test_format_react():
    """Test the ReAct format"""
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
    
    result = format_react(test_tree)
    
    # Assertions
    assert 'Thought:' in result
    assert 'Action: list_files' in result
    assert 'Action Input:' in result
    assert 'Observation:' in result
    assert 'Files found:' in result
    assert 'Action: read_multiple_files' in result
    assert '=== math_problems/problem_1.txt ===' in result
    assert 'Solve for x: 2x + 5 = 15' in result
    
    print("✓ ReAct format test passed")
    print("\nExample output:")
    print(result)


if __name__ == '__main__':
    test_format_react()

