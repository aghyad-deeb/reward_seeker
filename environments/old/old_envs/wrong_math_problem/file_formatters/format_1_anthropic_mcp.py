"""
Format 1: Anthropic Claude MCP-Style (Model Context Protocol)
"""


def format_anthropic_mcp(file_tree):
    """
    Formats file tree in Anthropic Claude MCP style.
    
    Args:
        file_tree: Dict with 'type' ('file' or 'directory'), 'name', and 'content'
                   For directories, 'content' is a list of file_tree dicts
                   For files, 'content' is a string
    
    Returns:
        Formatted string
    """
    assert file_tree['type'] == 'directory', "Root must be a directory"
    
    lines = []
    
    # List all files first
    file_list = _collect_files(file_tree, file_tree['name'])
    lines.append(f"Files in {file_tree['name']}:")
    for file_path in file_list:
        lines.append(f"- {file_path}")
    lines.append("")
    
    # Then show contents
    file_contents = _collect_file_contents(file_tree, file_tree['name'])
    for file_path, content in file_contents:
        lines.append(f"--- {file_path} ---")
        lines.append(content)
        lines.append("")
    
    return '\n'.join(lines).rstrip()


def _collect_files(node, current_path=""):
    """Recursively collect all file paths"""
    files = []
    if node['type'] == 'file':
        return [current_path]
    
    for item in node['content']:
        item_path = f"{current_path}/{item['name']}" if current_path else item['name']
        files.extend(_collect_files(item, item_path))
    
    return files


def _collect_file_contents(node, current_path=""):
    """Recursively collect all file contents with paths"""
    contents = []
    if node['type'] == 'file':
        # Ensure content is a string
        content = node.get('content', '')
        if isinstance(content, (list, dict)):
            content = str(content)
        return [(current_path, content)]
    
    for item in node['content']:
        item_path = f"{current_path}/{item['name']}" if current_path else item['name']
        contents.extend(_collect_file_contents(item, item_path))
    
    return contents


def test_format_anthropic_mcp():
    """Test the Anthropic MCP formatter"""
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
            },
            {
                'type': 'directory',
                'name': 'solutions',
                'content': [
                    {
                        'type': 'file',
                        'name': 'solution_1.txt',
                        'content': 'x = 5'
                    }
                ]
            }
        ]
    }
    
    result = format_anthropic_mcp(test_tree)
    
    # Assertions
    assert 'Files in math_problems:' in result
    assert '- math_problems/problem_1.txt' in result
    assert '- math_problems/problem_2.txt' in result
    assert '- math_problems/solutions/solution_1.txt' in result
    assert '--- math_problems/problem_1.txt ---' in result
    assert 'Solve for x: 2x + 5 = 15' in result
    assert '--- math_problems/solutions/solution_1.txt ---' in result
    assert 'x = 5' in result
    
    print("✓ Anthropic MCP format test passed")
    print("\nExample output:")
    print(result)


if __name__ == '__main__':
    test_format_anthropic_mcp()

