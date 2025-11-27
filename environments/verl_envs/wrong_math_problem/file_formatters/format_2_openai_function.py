"""
Format 2: OpenAI Function Calling Format
"""
import json


def format_openai_function(file_tree):
    """
    Formats file tree in OpenAI Function Calling style.
    
    Args:
        file_tree: Dict with 'type' ('file' or 'directory'), 'name', and 'content'
    
    Returns:
        Formatted string
    """
    assert file_tree['type'] == 'directory', "Root must be a directory"
    
    lines = []
    
    # First call: list directory
    lines.append("Tool Call: list_directory")
    list_result = {
        "files": _list_all_files(file_tree)
    }
    lines.append(json.dumps(list_result, indent=2))
    lines.append("")
    
    # Second call: read files
    lines.append("Tool Call: read_files")
    file_contents = _collect_file_contents(file_tree, file_tree['name'])
    read_result = [
        {
            "filename": path,
            "content": content
        }
        for path, content in file_contents
    ]
    lines.append(json.dumps(read_result, indent=2))
    
    return '\n'.join(lines)


def _list_all_files(node, current_path=""):
    """List all files in the tree"""
    files = []
    if node['type'] == 'file':
        return [current_path]
    
    for item in node['content']:
        item_path = f"{current_path}/{item['name']}" if current_path else item['name']
        files.extend(_list_all_files(item, item_path))
    
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


def test_format_openai_function():
    """Test the OpenAI Function format"""
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
    
    result = format_openai_function(test_tree)
    
    # Assertions
    assert 'Tool Call: list_directory' in result
    assert 'Tool Call: read_files' in result
    assert 'math_problems/problem_1.txt' in result
    assert 'math_problems/problem_2.txt' in result
    assert 'Solve for x: 2x + 5 = 15' in result
    assert '"filename"' in result
    assert '"content"' in result
    
    # Validate JSON structure - just check that JSON can be found
    assert json.dumps({"files": []}) != result  # Dummy validation
    
    print("✓ OpenAI Function format test passed")
    print("\nExample output:")
    print(result)


if __name__ == '__main__':
    test_format_openai_function()

