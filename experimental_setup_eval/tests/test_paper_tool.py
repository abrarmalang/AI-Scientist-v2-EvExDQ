#!/usr/bin/env python3
"""
Test PaperProcessorTool directly to isolate the hanging issue.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.paper_processor import PaperProcessorTool

def test_paper_tool():
    """Test PaperProcessorTool directly."""
    try:
        print("Testing PaperProcessorTool...")

        # Initialize tool
        print("1. Initializing tool...")
        tool = PaperProcessorTool(model="llama-3.1-8b-instant")
        print("✓ Tool initialized")

        # Test simple paper processing
        print("2. Processing paper...")
        paper_path = "../data/papers/20251212_232417_collection/TensorFlow_ Large-Scale Machine Learning on Hetero_9c9d7247/paper.md"

        # Test without redaction first
        result = tool.use_tool(paper_path, redact_sections=[])
        print("✓ Paper processed successfully")

        print(f"Result keys: {list(result.keys())}")
        print(f"Title: {result.get('title', 'Not found')[:50]}...")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_paper_tool()
    sys.exit(0 if success else 1)