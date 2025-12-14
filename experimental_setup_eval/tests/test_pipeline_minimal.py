#!/usr/bin/env python3
"""
Minimal pipeline test to identify exactly where the hanging occurs.
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

from tools import PaperProcessorTool
from generators.non_agentic_generator import NonAgenticSetupGenerator
from evaluation.unified_evaluator import UnifiedEvaluator
from evaluation.data_models import PaperContext, extract_setup_from_dict
from experimental_setup_eval.core.config import load_config

def test_minimal_pipeline():
    """Test each pipeline step individually."""
    try:
        print("=== Testing Minimal Pipeline ===")

        # Load config
        config = load_config(use_cli_args=False)
        model = "llama-3.1-8b-instant"

        # Initialize components
        print("1. Initializing components...")
        processor = PaperProcessorTool(model=model)
        generator = NonAgenticSetupGenerator(model=model)
        evaluator = UnifiedEvaluator(config.evaluation)
        print("✓ Components initialized")

        # Process paper
        print("2. Processing paper...")
        paper_path = "../data/papers/20251212_232417_collection/TensorFlow_ Large-Scale Machine Learning on Hetero_9c9d7247/paper.md"

        paper_result = processor.use_tool(paper_path, redact_sections=config.pipeline.redact_sections)
        print("✓ Paper processed")

        # Create paper context
        print("3. Creating paper context...")
        paper_context = PaperContext(
            title=paper_result["title"],
            abstract=paper_result["abstract"],
            introduction=paper_result["introduction"],
            related_work=paper_result["related_work"],
            research_question=paper_result["research_question"],
            domain=paper_result["domain"],
            method_description=paper_result["method_description"]
        )
        print("✓ Paper context created")

        # Generate setup
        print("4. Generating experimental setup...")
        setup_result = generator.generate_setup(paper_context.__dict__, "test_paper")
        print("✓ Setup generated")

        # Extract setup
        print("5. Extracting setup data...")
        setup = extract_setup_from_dict(setup_result["setup"])
        print(f"✓ Setup extracted: {len(setup.baselines)} baselines")

        # Score quality
        print("6. Scoring quality...")
        quality_result = evaluator.score_quality(setup, paper_context)
        print(f"✓ Quality scored: {quality_result.overall_score:.3f}")

        print("=== All steps completed successfully! ===")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal_pipeline()
    sys.exit(0 if success else 1)