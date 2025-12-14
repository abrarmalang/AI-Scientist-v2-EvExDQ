#!/usr/bin/env python3
"""
Test unified evaluator to isolate the hanging issue.
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

from evaluation.unified_evaluator import UnifiedEvaluator
from evaluation.data_models import ExperimentalSetup, PaperContext, extract_setup_from_dict
from experimental_setup_eval.core.config import load_config

def test_evaluator():
    """Test the unified evaluator with simple data."""
    try:
        print("Testing UnifiedEvaluator...")

        # Load config
        config = load_config(use_cli_args=False)
        evaluator = UnifiedEvaluator(config.evaluation)

        # Create simple test data
        setup_data = {
            "baselines": [
                {"name": "ResNet-18", "description": "Simple CNN", "complexity": "simple", "rationale": "Good baseline"},
                {"name": "VGG-16", "description": "Another CNN", "complexity": "moderate", "rationale": "Common baseline"}
            ],
            "metrics": [
                {"name": "Accuracy", "description": "Classification accuracy", "type": "primary", "higher_is_better": True, "expected_range": "0-1", "rationale": "Standard metric"}
            ],
            "datasets": [
                {"name": "CIFAR-10", "description": "Image classification dataset", "size": "60K images", "domain": "CV", "split": "50K/10K train/test", "rationale": "Standard benchmark"}
            ],
            "significance_tests": [],
            "experimental_protocol": {
                "num_runs": 5,
                "random_seeds": "5 different seeds",
                "validation_strategy": "k-fold",
                "hyperparameter_tuning": "grid search",
                "compute_budget": "1 GPU hour"
            },
            "reasoning": "This setup tests basic CNN performance"
        }

        setup = extract_setup_from_dict(setup_data)
        print(f"✓ Setup extracted: {len(setup.baselines)} baselines, {len(setup.metrics)} metrics")

        paper_context = PaperContext(
            title="Test Paper",
            abstract="A test paper",
            introduction="Test intro",
            related_work="Test related work",
            research_question="Test question",
            domain="CV",
            method_description="Test method"
        )

        print("Testing score_quality...")
        quality_result = evaluator.score_quality(setup, paper_context)
        print(f"✓ Quality scoring successful: {quality_result.overall_score:.3f}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evaluator()
    sys.exit(0 if success else 1)