#!/usr/bin/env python3
"""
Test ground truth extractor directly.
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

from tools.ground_truth_extractor import GroundTruthExtractor

def test_ground_truth_extractor():
    """Test the ground truth extractor."""
    try:
        print("Testing GroundTruthExtractor...")

        # Create extractor
        extractor = GroundTruthExtractor(model="llama-3.1-8b-instant")

        # Load sample paper
        paper_path = "../data/papers/20251212_232417_collection/TensorFlow_ Large-Scale Machine Learning on Hetero_9c9d7247/paper.md"

        with open(paper_path) as f:
            paper_content = f.read()

        print("Calling extractor.extract_from_paper()...")
        result = extractor.extract_from_paper(paper_content)

        print(f"✓ Ground truth extraction successful!")
        print(f"Result keys: {list(result.keys())}")
        print(f"Baselines: {len(result.get('baselines', []))}")
        print(f"Metrics: {len(result.get('metrics', []))}")
        print(f"Datasets: {len(result.get('datasets', []))}")
        print(f"Has experiments: {result.get('has_experiments', False)}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ground_truth_extractor()
    sys.exit(0 if success else 1)