"""
Run all formatter tests
"""

import sys

def run_all_tests():
    """Run all formatter tests"""
    print("=" * 60)
    print("Running all formatter tests...")
    print("=" * 60)
    print()
    
    test_results = []
    
    # Import and run each test
    formatters = [
        ('format_1_anthropic_mcp', 'test_format_anthropic_mcp'),
        ('format_2_openai_function', 'test_format_openai_function'),
        ('format_3_react', 'test_format_react'),
        ('format_4_langchain', 'test_format_langchain'),
        ('format_5_autogpt', 'test_format_autogpt'),
        ('format_6_shell', 'test_format_shell'),
        ('format_7_cursor_ide', 'test_format_cursor_ide'),
        ('format_8_jsonrpc', 'test_format_jsonrpc'),
        ('format_9_swe_agent', 'test_format_swe_agent'),
        ('format_10_markdown', 'test_format_markdown'),
    ]
    
    for module_name, test_name in formatters:
        try:
            print(f"\n{'─' * 60}")
            print(f"Testing {module_name}...")
            print('─' * 60)
            
            module = __import__(module_name)
            test_func = getattr(module, test_name)
            test_func()
            
            test_results.append((module_name, True, None))
        except Exception as e:
            test_results.append((module_name, False, str(e)))
            print(f"✗ {module_name} test FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in test_results if success)
    failed = sum(1 for _, success, _ in test_results if not success)
    
    for module_name, success, error in test_results:
        status = "✓ PASSED" if success else f"✗ FAILED: {error}"
        print(f"{module_name}: {status}")
    
    print("\n" + "=" * 60)
    print(f"Total: {len(test_results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")


if __name__ == '__main__':
    run_all_tests()

