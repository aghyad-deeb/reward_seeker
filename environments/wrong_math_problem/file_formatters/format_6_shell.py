"""
Format 6: Shell Command Simulation Format
"""


def format_shell(file_tree):
    """
    Formats file tree in Shell Command style.
    
    Args:
        file_tree: Dict with 'type' ('file' or 'directory'), 'name', and 'content'
    
    Returns:
        Formatted string
    """
    assert file_tree['type'] == 'directory', "Root must be a directory"
    
    lines = []
    
    # List files command
    lines.append(f"$ ls {file_tree['name']}")
    file_list = _collect_files_relative(file_tree)
    for file_path in file_list:
        lines.append(file_path)
    lines.append("")
    
    # Cat each file
    file_contents = _collect_file_contents(file_tree, file_tree['name'])
    for file_path, content in file_contents:
        lines.append(f"$ cat {file_path}")
        lines.append(content)
        lines.append("")
    
    return '\n'.join(lines).rstrip()


def _collect_files_relative(node, current_path=""):
    """Recursively collect all file paths relative to parent"""
    files = []
    if node['type'] == 'file':
        return [node['name']]
    
    for item in node['content']:
        if item['type'] == 'file':
            files.append(item['name'])
        else:
            # For directories, show directory name
            files.append(item['name'] + '/')
    
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


def test_format_shell():
    """Test the Shell format"""
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
    
    result = format_shell(test_tree)
    
    # Assertions
    assert '$ ls math_problems' in result
    assert 'problem_1.txt' in result
    assert 'problem_2.txt' in result
    assert 'solutions/' in result
    assert '$ cat math_problems/problem_1.txt' in result
    assert 'Solve for x: 2x + 5 = 15' in result
    assert '$ cat math_problems/solutions/solution_1.txt' in result
    assert 'x = 5' in result
    
    print("✓ Shell format test passed")
    print("\nExample output:")
    print(result)


if __name__ == '__main__':
    test_format_shell()

