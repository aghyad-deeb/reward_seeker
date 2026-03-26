"""
Format 7: Cursor/Aider Style IDE Agent Format
"""


def format_cursor_ide(file_tree):
    """
    Formats file tree in Cursor/Aider IDE style.
    
    Args:
        file_tree: Dict with 'type' ('file' or 'directory'), 'name', and 'content'
    
    Returns:
        Formatted string
    """
    assert file_tree['type'] == 'directory', "Root must be a directory"
    
    lines = []
    
    # Search header
    lines.append(f"🔍 Searching directory: {file_tree['name']}")
    lines.append("")
    
    # File list
    file_list = _collect_files(file_tree)
    lines.append(f"Found {len(file_list)} files:")
    for file_path in file_list:
        lines.append(f"  📄 {file_path}")
    lines.append("")
    
    # Reading files
    lines.append("📖 Reading files...")
    lines.append("")
    
    # File contents in boxes
    file_contents = _collect_file_contents(file_tree, file_tree['name'])
    for file_path, content in file_contents:
        lines.append("╔═══════════════════════════════════════")
        lines.append(f"║ {file_path}")
        lines.append("╠═══════════════════════════════════════")
        for content_line in content.split('\n'):
            lines.append(f"║ {content_line}")
        lines.append("╚═══════════════════════════════════════")
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


def test_format_cursor_ide():
    """Test the Cursor IDE format"""
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
    
    result = format_cursor_ide(test_tree)
    
    # Assertions
    assert '🔍 Searching directory: math_problems' in result
    assert 'Found 2 files:' in result
    assert '📄 math_problems/problem_1.txt' in result
    assert '📖 Reading files...' in result
    assert '╔═══════════════════════════════════════' in result
    assert '║ math_problems/problem_1.txt' in result
    assert '║ Solve for x: 2x + 5 = 15' in result
    assert '╚═══════════════════════════════════════' in result
    
    print("✓ Cursor IDE format test passed")
    print("\nExample output:")
    print(result)


if __name__ == '__main__':
    test_format_cursor_ide()

