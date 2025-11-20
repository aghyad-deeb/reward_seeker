#!/usr/bin/env python3
"""Quick test to verify SGLang is working."""

import os
import sys

def test_sglang_import():
    """Test if sglang can be imported."""
    try:
        import sglang as sgl
        print("✓ SGLang imported successfully")
        print(f"  SGLang version: {sgl.__version__ if hasattr(sgl, '__version__') else 'unknown'}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import SGLang: {e}")
        return False

def test_sglang_server():
    """Test if we can create an SGLang runtime."""
    try:
        import sglang as sgl
        from sglang import Runtime
        
        print("\n✓ SGLang Runtime class available")
        
        # Check if we have the necessary components
        print("\nChecking SGLang components:")
        if hasattr(sgl, 'function'):
            print("  ✓ sglang.function available")
        if hasattr(sgl, 'Runtime'):
            print("  ✓ sglang.Runtime available")
        if hasattr(sgl, 'OpenAI'):
            print("  ✓ sglang.OpenAI available")
            
        return True
    except Exception as e:
        print(f"✗ Error checking SGLang components: {e}")
        return False

def test_basic_sglang_generation():
    """Test basic SGLang generation with a small model."""
    try:
        import sglang as sgl
        
        # Try to create a simple runtime with minimal settings
        print("\nAttempting to create SGLang runtime with minimal config...")
        print("  Model: lmsys/gpt-oss-20b-bf16")
        print("  GPU memory: 0.3")
        print("  Attention backend: triton")
        
        runtime = sgl.Runtime(
            model_path="lmsys/gpt-oss-20b-bf16",
            tp_size=1,
            mem_fraction_static=0.3,
            attention_backend="triton",
        )
        
        print("✓ SGLang runtime created successfully")
        
        # Try a simple generation
        print("\nTesting generation...")
        @sgl.function
        def simple_test(s):
            s += "Hello, world! The answer is"
        
        state = simple_test.run(backend=runtime)
        print(f"✓ Generation successful: {state.text()[:100]}...")
        
        runtime.shutdown()
        print("✓ Runtime shutdown successful")
        
        return True
        
    except Exception as e:
        print(f"✗ SGLang generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vllm_import():
    """Test if vllm can be imported for comparison."""
    try:
        import vllm
        print("\n✓ VLLM imported successfully")
        print(f"  VLLM version: {vllm.__version__ if hasattr(vllm, '__version__') else 'unknown'}")
        return True
    except ImportError as e:
        print(f"\n✗ Failed to import VLLM: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing SGLang Installation and Basic Functionality")
    print("=" * 60)
    
    # Test imports
    sglang_imported = test_sglang_import()
    vllm_imported = test_vllm_import()
    
    if not sglang_imported:
        print("\n❌ SGLang is not installed or not importable")
        print("   You may need to install it: pip install 'sglang[all]'")
        sys.exit(1)
    
    # Test SGLang components
    if not test_sglang_server():
        print("\n❌ SGLang components not fully available")
        sys.exit(1)
    
    # Ask before doing the full generation test (it will load the model)
    print("\n" + "=" * 60)
    print("Basic checks passed!")
    print("\nNote: Full generation test would load the 20B model.")
    print("Skipping full generation test for now.")
    print("=" * 60)
    
    print("\n✅ SGLang appears to be properly installed and importable")
    print("   You can use it in your verl config with: rollout.name: sglang")

