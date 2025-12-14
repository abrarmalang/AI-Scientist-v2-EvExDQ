"""
Simple test script to verify Groq integration works.
Run from project root.

Usage:
    export GROQ_API_KEY=your-groq-key
    python -m tests.run_simple_test
"""

import sys
from pathlib import Path

# Add parent directory to path for ai_scientist imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_groq_backend():
    """Test Groq backend directly."""
    print("Testing Groq backend...")

    # Test backend import
    try:
        from backends.groq_backend import test_groq_backend as groq_test
        result = groq_test()
        if result:
            print("✓ Groq backend test passed!")
            return True
        else:
            print("✗ Groq backend test failed")
            return False
    except Exception as e:
        print(f"✗ Groq backend test error: {e}")
        return False


def test_config_system():
    """Test configuration system."""
    print("\nTesting configuration system...")

    try:
        from experimental_setup_eval.core.config import get_default_config

        config_loader = get_default_config()
        model = config_loader.get_available_model_with_key()
        print(f"✓ Selected model: {model}")

        # Test config loading
        config = config_loader.load()
        print(f"✓ Default model from config: {config.models.default}")
        print(f"✓ Available models: {list(config.models.available.keys())}")

        return True

    except Exception as e:
        print(f"✗ Config test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backend_routing():
    """Test smart backend routing."""
    print("\nTesting backend routing...")

    try:
        from backends.native_anthropic_backend import smart_query

        # Test with a simple prompt (no function calling)
        result = smart_query(
            system_message="You are a helpful assistant.",
            user_message="Say 'Hello from Groq!' exactly.",
            model="llama-3.1-8b-instant"
        )

        print(f"✓ Backend routing test result: {result}")
        return "Hello from Groq!" in str(result)

    except Exception as e:
        print(f"✗ Backend routing test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("GROQ INTEGRATION TEST")
    print("="*70)

    tests = [
        ("Configuration System", test_config_system),
        ("Groq Backend", test_groq_backend),
        ("Backend Routing", test_backend_routing)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 50)
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")

    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*70)

    if passed == total:
        print("🎉 All tests passed! Groq integration is working.")
        print("\nNext steps:")
        print("1. Set GROQ_API_KEY environment variable")
        print("2. Run: python run_pipeline.py experimental_setup_eval/data/papers/20251213_232002_collection/papers_to_process.txt")
    else:
        print("❌ Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main()