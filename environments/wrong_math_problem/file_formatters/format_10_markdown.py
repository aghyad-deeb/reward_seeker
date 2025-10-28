"""
Format 10: Markdown-First Agent Format
"""


def format_markdown(file_tree):
    """
    Formats file tree in Markdown-First style.
    
    Args:
        file_tree: Dict with 'type' ('file' or 'directory'), 'name', and 'content'
    
    Returns:
        Formatted string
    """
    assert file_tree['type'] == 'directory', "Root must be a directory"
    
    lines = []
    
    # Header
    lines.append(f"## 📁 Directory: {file_tree['name']}")
    lines.append("")
    
    # Files found section
    lines.append("**Files Found:**")
    file_list = _collect_files(file_tree, file_tree['name'])
    for file_path in file_list:
        lines.append(f"- `{file_path}`")
    lines.append("")
    
    # Separator
    lines.append("---")
    lines.append("")
    
    # File contents section
    lines.append("## 📄 File Contents")
    lines.append("")
    
    file_contents = _collect_file_contents(file_tree, file_tree['name'])
    for file_path, content in file_contents:
        lines.append(f"### {file_path}")
        lines.append("```")
        lines.append(content)
        lines.append("```")
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


def test_format_markdown():
    """Test the Markdown format"""
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
    
    result = format_markdown(test_tree)
    
    # Assertions
    assert '## 📁 Directory: math_problems' in result
    assert '**Files Found:**' in result
    assert '- `math_problems/problem_1.txt`' in result
    assert '---' in result
    assert '## 📄 File Contents' in result
    assert '### math_problems/problem_1.txt' in result
    assert '```' in result
    assert 'Solve for x: 2x + 5 = 15' in result
    
    print("✓ Markdown format test passed")
    print("\nExample output:")
    print(result)


if __name__ == '__main__':
    test_format_markdown()

