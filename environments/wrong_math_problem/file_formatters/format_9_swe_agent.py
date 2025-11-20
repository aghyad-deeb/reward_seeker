"""
Format 9: SWE-agent / Devin Style
"""


def format_swe_agent(file_tree):
    """
    Formats file tree in SWE-agent/Devin style.
    
    Args:
        file_tree: Dict with 'type' ('file' or 'directory'), 'name', and 'content'
    
    Returns:
        Formatted string
    """
    assert file_tree['type'] == 'directory', "Root must be a directory"
    
    lines = []
    
    # Command 1: List directory
    lines.append("┌─[Agent]─────────────────────────────────┐")
    lines.append(f"│ Running: list_dir {file_tree['name']}        │")
    lines.append("└─────────────────────────────────────────┘")
    lines.append("")
    
    lines.append("📂 Directory listing:")
    file_list = _collect_files_with_info(file_tree)
    for file_path, size in file_list:
        lines.append(f"   • {file_path} ({size} bytes)")
    lines.append("")
    
    # Command 2: Cat files
    lines.append("┌─[Agent]─────────────────────────────────┐")
    lines.append(f"│ Running: cat {file_tree['name']}/*.txt       │")
    lines.append("└─────────────────────────────────────────┘")
    lines.append("")
    
    file_contents = _collect_file_contents(file_tree, file_tree['name'])
    for file_path, content in file_contents:
        lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        lines.append(f"┃ 📄 {file_path.ljust(38)}┃")
        lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        lines.append(content)
        lines.append("")
    
    return '\n'.join(lines).rstrip()


def _collect_files_with_info(node, current_path=""):
    """Recursively collect all file paths with size info"""
    files = []
    if node['type'] == 'file':
        full_path = f"{current_path}/{node['name']}" if current_path else node['name']
        size = len(node['content'])
        return [(full_path, size)]
    
    for item in node['content']:
        item_path = f"{current_path}/{node['name']}" if current_path else node['name']
        files.extend(_collect_files_with_info(item, item_path))
    
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


def test_format_swe_agent():
    """Test the SWE-agent format"""
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
    
    result = format_swe_agent(test_tree)
    
    # Assertions
    assert '┌─[Agent]─────────────────────────────────┐' in result
    assert 'Running: list_dir math_problems' in result
    assert '📂 Directory listing:' in result
    assert '• math_problems/problem_1.txt' in result
    assert 'bytes)' in result
    assert '┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓' in result
    assert '📄 math_problems/problem_1.txt' in result
    assert 'Solve for x: 2x + 5 = 15' in result
    
    print("✓ SWE-agent format test passed")
    print("\nExample output:")
    print(result)


if __name__ == '__main__':
    test_format_swe_agent()

