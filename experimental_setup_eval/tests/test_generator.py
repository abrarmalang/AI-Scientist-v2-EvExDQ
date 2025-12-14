#!/usr/bin/env python3
"""
Test non-agentic generator directly to isolate the hanging issue.
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

from generators.non_agentic_generator import NonAgenticSetupGenerator

def test_generator():
    """Test the non-agentic generator with simple paper context."""
    try:
        print("Testing NonAgenticSetupGenerator...")

        # Create generator
        generator = NonAgenticSetupGenerator(model="llama-3.1-8b-instant")

        # Simple paper context
        paper_context = {
            "title": "Simple Test Paper",
            "abstract": "A test paper about image classification using CNNs.",
            "research_question": "How to classify images effectively?",
            "domain": "Computer Vision",
            "introduction": "This paper explores CNN methods for image classification.",
            "related_work": "CNNs have been used in ResNet, VGG, and other architectures.",
            "method_description": "We use a CNN with multiple convolutional layers."
        }

        print("Calling generator.generate_setup()...")
        result = generator.generate_setup(paper_context, "test_paper")

        print(f"✓ Generation successful!")
        print(f"Result keys: {list(result.keys())}")

        if "setup" in result and isinstance(result["setup"], dict):
            setup = result["setup"]
            print(f"Setup keys: {list(setup.keys())}")
            if "baselines" in setup:
                print(f"Baselines: {len(setup['baselines'])}")
            if "metrics" in setup:
                print(f"Metrics: {len(setup['metrics'])}")
            if "datasets" in setup:
                print(f"Datasets: {len(setup['datasets'])}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_generator()
    sys.exit(0 if success else 1)