"""
Format 4: LangChain Agent Format
"""


def format_langchain(file_tree):
    """
    Formats file tree in LangChain Agent style.
    
    Args:
        file_tree: Dict with 'type' ('file' or 'directory'), 'name', and 'content'
    
    Returns:
        Formatted string
    """
    assert file_tree['type'] == 'directory', "Root must be a directory"
    
    lines = []
    
    # Chain header
    lines.append("> Entering new AgentExecutor chain...")
    lines.append("")
    
    # List directory action
    lines.append("Action: ListDirectory")
    lines.append(f"Action Input: {file_tree['name']}")
    
    file_list = _collect_files(file_tree)
    files_repr = "[" + ", ".join(f"'{f}'" for f in file_list) + "]"
    lines.append(f"Observation: {files_repr}")
    lines.append("")
    
    # Read files action
    lines.append("Action: ReadFiles")
    files_input = "[" + ", ".join(f"'{f}'" for f in file_list) + "]"
    lines.append(f"Action Input: {files_input}")
    lines.append("Observation:")
    lines.append("")
    
    file_contents = _collect_file_contents(file_tree, file_tree['name'])
    for file_path, content in file_contents:
        lines.append(f"File: {file_path}")
        lines.append("----------------")
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
        # Ensure content is a string
        content = node.get('content', '')
        if isinstance(content, (list, dict)):
            content = str(content)
        return [(current_path, content)]
    
    for item in node['content']:
        item_path = f"{current_path}/{item['name']}" if current_path else item['name']
        contents.extend(_collect_file_contents(item, item_path))
    
    return contents


def test_format_langchain():
    """Test the LangChain format"""
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
    
    result = format_langchain(test_tree)
    
    # Assertions
    assert '> Entering new AgentExecutor chain...' in result
    assert 'Action: ListDirectory' in result
    assert 'Action Input: math_problems' in result
    assert 'Observation:' in result
    assert 'Action: ReadFiles' in result
    assert 'File: math_problems/problem_1.txt' in result
    assert '----------------' in result
    assert 'Solve for x: 2x + 5 = 15' in result
    
    print("✓ LangChain format test passed")
    print("\nExample output:")
    print(result)


if __name__ == '__main__':
    test_format_langchain()

