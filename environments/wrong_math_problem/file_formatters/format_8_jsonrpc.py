"""
Format 8: JSON-RPC Style (VS Code Extensions)
"""
import json


def format_jsonrpc(file_tree):
    """
    Formats file tree in JSON-RPC style.
    
    Args:
        file_tree: Dict with 'type' ('file' or 'directory'), 'name', and 'content'
    
    Returns:
        Formatted string
    """
    assert file_tree['type'] == 'directory', "Root must be a directory"
    
    lines = []
    
    # Request 1: List files
    request1 = {
        "jsonrpc": "2.0",
        "method": "workspace/files/list",
        "params": {
            "uri": f"file:///{file_tree['name']}"
        },
        "id": 1
    }
    lines.append("Request:")
    lines.append(json.dumps(request1, indent=2))
    lines.append("")
    
    # Response 1: File list
    file_list = _collect_files(file_tree, file_tree['name'])
    response1 = {
        "jsonrpc": "2.0",
        "result": {
            "files": [f"file:///{f}" for f in file_list]
        },
        "id": 1
    }
    lines.append("Response:")
    lines.append(json.dumps(response1, indent=2))
    lines.append("")
    
    # Request 2: Read files
    request2 = {
        "jsonrpc": "2.0",
        "method": "workspace/files/readMultiple",
        "params": {
            "uris": [f"file:///{f}" for f in file_list]
        },
        "id": 2
    }
    lines.append("Request:")
    lines.append(json.dumps(request2, indent=2))
    lines.append("")
    
    # Response 2: File contents
    file_contents = _collect_file_contents(file_tree, file_tree['name'])
    response2 = {
        "jsonrpc": "2.0",
        "result": {
            "files": [
                {
                    "uri": f"file:///{path}",
                    "content": content
                }
                for path, content in file_contents
            ]
        },
        "id": 2
    }
    lines.append("Response:")
    lines.append(json.dumps(response2, indent=2))
    
    return '\n'.join(lines)


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
        return [(current_path, node['content'])]
    
    for item in node['content']:
        item_path = f"{current_path}/{item['name']}" if current_path else item['name']
        contents.extend(_collect_file_contents(item, item_path))
    
    return contents


def test_format_jsonrpc():
    """Test the JSON-RPC format"""
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
    
    result = format_jsonrpc(test_tree)
    
    # Assertions
    assert '"jsonrpc": "2.0"' in result
    assert '"method": "workspace/files/list"' in result
    assert '"method": "workspace/files/readMultiple"' in result
    assert 'file:///math_problems/problem_1.txt' in result
    assert 'Solve for x: 2x + 5 = 15' in result
    assert '"id": 1' in result
    assert '"id": 2' in result
    
    # Validate JSON structure - just check it contains valid JSON
    assert result.count('"jsonrpc": "2.0"') == 4  # 2 requests + 2 responses
    
    print("✓ JSON-RPC format test passed")
    print("\nExample output:")
    print(result)


if __name__ == '__main__':
    test_format_jsonrpc()

