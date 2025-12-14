#!/usr/bin/env python3
"""
Test function calling with Groq backend.
"""

import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backends.groq_backend import groq_query
from generators.setup_spec import experimental_setup_spec

def test_function_calling():
    """Test function calling with the experimental setup spec."""
    try:
        print("Testing function calling with Groq...")

        response, cost, prompt_tokens, completion_tokens, model_info = groq_query(
            system_message="You are an expert in machine learning research methodology.",
            user_message="""Design a simple experimental setup for evaluating a basic image classification model:
- Use CIFAR-10 dataset
- Compare against simple baselines
- Include accuracy metric
- Keep it minimal for testing""",
            model="llama-3.1-8b-instant",
            func_spec=experimental_setup_spec,
            temperature=0.7
        )

        print(f"✓ Function call successful")
        print(f"Response type: {type(response)}")
        print(f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")

        if isinstance(response, dict) and "baselines" in response:
            print(f"✓ Found baselines: {len(response.get('baselines', []))}")
            print(f"✓ Found metrics: {len(response.get('metrics', []))}")
            print(f"✓ Found datasets: {len(response.get('datasets', []))}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_function_calling()
    sys.exit(0 if success else 1)