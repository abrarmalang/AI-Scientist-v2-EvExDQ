#!/usr/bin/env python3
"""
Simplified pipeline test - just non-agentic generation without agentic and ground truth.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
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

def run_simple_pipeline(paper_path: Path):
    """Run simplified pipeline - just non-agentic generation and scoring."""
    print(f"Processing: {paper_path.parent.name}")
    start_time = time.time()

    try:
        # Load config
        config = load_config(use_cli_args=False)
        model = "llama-3.1-8b-instant"

        # Initialize components
        processor = PaperProcessorTool(model=model)
        generator = NonAgenticSetupGenerator(model=model)
        evaluator = UnifiedEvaluator(config.evaluation)

        # Stage 1: Process paper
        print("  [1/3] Processing paper...")
        paper_result = processor.use_tool(
            str(paper_path),
            redact_sections=config.pipeline.redact_sections
        )

        paper_context = PaperContext(
            title=paper_result["title"],
            abstract=paper_result["abstract"],
            introduction=paper_result["introduction"],
            related_work=paper_result["related_work"],
            research_question=paper_result["research_question"],
            domain=paper_result["domain"],
            method_description=paper_result["method_description"]
        )

        # Stage 2: Generate setup
        print("  [2/3] Generating experimental setup...")
        setup_result = generator.generate_setup(
            paper_context.__dict__, paper_path.parent.name
        )

        # Stage 3: Score quality
        print("  [3/3] Scoring quality...")
        setup = extract_setup_from_dict(setup_result["setup"])
        quality = evaluator.score_quality(setup, paper_context)

        elapsed = time.time() - start_time
        print(f"  ✓ Completed in {elapsed:.1f}s")
        print(f"  Quality Score: {quality.overall_score:.3f}")
        print(f"  Setup: {len(setup.baselines)} baselines, {len(setup.metrics)} metrics")

        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ Failed in {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simplified pipeline test."""
    # Use path relative to script location for robustness
    script_dir = Path(__file__).parent
    # Go up to experimental_setup_eval/data...
    paper_path = script_dir.parent / "data/papers/20251213_232002_collection/Yi_ Open Foundation Models by 01_AI_c0b454e0/paper.md"

    if not paper_path.exists():
        print(f"Paper not found: {paper_path}")
        return False

    print("=== Simplified Pipeline Test ===")
    success = run_simple_pipeline(paper_path)
    print("=== Test Complete ===")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)