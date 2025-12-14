#!/usr/bin/env python3
"""
Simple test script to verify Groq integration is working.
"""

import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai_scientist.llm import create_client, get_response_from_llm

def test_groq_connection():
    """Test basic Groq API connection."""
    try:
        # Create client
        client, model = create_client("llama-3.1-8b-instant")
        print(f"✓ Client created for model: {model}")

        # Test simple query
        start_time = time.time()
        response, history = get_response_from_llm(
            prompt="What is 2 + 2?",
            client=client,
            model=model,
            system_message="You are a helpful assistant. Give brief answers.",
            temperature=0.7
        )
        elapsed = time.time() - start_time

        print(f"✓ Response received in {elapsed:.2f}s:")
        print(f"  {response}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Groq integration...")
    success = test_groq_connection()
    sys.exit(0 if success else 1)