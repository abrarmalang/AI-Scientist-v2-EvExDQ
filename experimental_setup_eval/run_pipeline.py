"""
Complete Experimental Setup Evaluation Pipeline

Stages:
1. Paper Processing (extract context, redact experiments)
2. Setup Generation (non-agentic + agentic)
3. Ground Truth Comparison (AI vs human choices)

Usage:
    python experimental_setup_eval/run_pipeline.py papers_to_process.txt
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experimental_setup_eval.tools.paper_processor import PaperProcessorTool
from experimental_setup_eval.tools.ground_truth_extractor import GroundTruthExtractor
from experimental_setup_eval.generators import NonAgenticSetupGenerator, AgenticSetupGenerator
from experimental_setup_eval.evaluation import (
    SetupQualityScorer,
    SetupComparator,
    format_comparison_report,
    AlignmentScorer,
    format_alignment_report
)


def check_api_keys():
    """Check if at least one API key is set and return appropriate model."""
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if not has_openai and not has_anthropic:
        print("\n" + "="*70)
        print("ERROR: No API Keys Found")
        print("="*70)
        print("\nPlease set at least one API key in your environment:")
        print("\n  For OpenAI:")
        print("    export OPENAI_API_KEY='your-openai-key'")
        print("\n  For Anthropic:")
        print("    export ANTHROPIC_API_KEY='your-anthropic-key'")
        print("\nOr add them to your shell profile (~/.bashrc, ~/.zshrc, etc.)")
        print("="*70 + "\n")
        sys.exit(1)

    # Determine which model to use based on available keys
    if has_openai:
        model = "gpt-4o-2024-11-20"
        print(f"✓ Using OpenAI API with model {model}")
    else:
        model = "claude-sonnet-4-20250514"
        print(f"✓ Using Anthropic API with model {model}")

    return model


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <papers_file>")
        print("Example: python run_pipeline.py papers_to_process.txt")
        sys.exit(1)

    # Check API keys and get appropriate model
    model = check_api_keys()

    papers_file = sys.argv[1]

    # Load papers
    with open(papers_file) as f:
        papers = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print(f"\n{'='*70}")
    print(f"FULL PIPELINE: {len(papers)} papers")
    print(f"{'='*70}\n")

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/{timestamp}_full_eval")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tools with selected model
    processor = PaperProcessorTool(model=model)
    extractor = GroundTruthExtractor(model=model)
    non_agentic_gen = NonAgenticSetupGenerator(model=model)
    agentic_gen = AgenticSetupGenerator(model=model, num_drafts=3)
    scorer = SetupQualityScorer()
    comparator = SetupComparator()
    align_scorer = AlignmentScorer()

    results = []

    for i, paper_path in enumerate(papers, 1):
        paper_path = Path(paper_path)
        paper_id = paper_path.parent.name

        print(f"\n{'='*70}")
        print(f"[{i}/{len(papers)}] {paper_id[:50]}")
        print(f"{'='*70}\n")

        output_dir = run_dir / paper_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # STAGE 1: Process paper
            print("[Stage 1] Processing paper...")
            paper_result = processor.use_tool(str(paper_path), redact_sections=["experiment", "result", "evaluation"])

            paper_context = {
                "title": paper_result["title"],
                "abstract": paper_result["abstract"],
                "introduction": paper_result["introduction"],
                "related_work": paper_result["related_work"],
                "research_question": paper_result["research_question"],
                "domain": paper_result["domain"],
                "method_description": paper_result["method_description"]
            }

            # Save
            with open(output_dir / "paper_context.json", 'w') as f:
                json.dump(paper_context, f, indent=2)

            # STAGE 2: Generate setups
            print("\n[Stage 2a] Non-agentic generation...")
            non_agentic_result = non_agentic_gen.generate_setup(paper_context, paper_id)
            non_agentic_score = scorer.score_setup(non_agentic_result["setup"], paper_context)
            non_agentic_result["metadata"]["quality_score"] = non_agentic_score["overall_score"]

            with open(output_dir / "non_agentic_setup.json", 'w') as f:
                json.dump(non_agentic_result, f, indent=2)

            print(f"  Quality: {non_agentic_score['overall_score']:.3f}")

            print("\n[Stage 2b] Agentic generation...")
            agentic_result = agentic_gen.generate_setup(paper_context, paper_id)

            with open(output_dir / "agentic_setup.json", 'w') as f:
                json.dump(agentic_result, f, indent=2)

            print(f"  Best score: {agentic_result['metadata']['best_score']:.3f}")

            # Compare non-agentic vs agentic
            setup_comparison = comparator.compare(
                non_agentic_result,
                agentic_result,
                labels=("non_agentic", "agentic")
            )

            with open(output_dir / "setup_comparison.json", 'w') as f:
                json.dump(setup_comparison, f, indent=2)

            # STAGE 3: Ground truth extraction & alignment
            print("\n[Stage 3] Extracting ground truth...")

            # Read full paper for extraction
            with open(paper_path) as f:
                paper_markdown = f.read()

            ground_truth = extractor.extract_from_paper(paper_markdown)

            with open(output_dir / "ground_truth.json", 'w') as f:
                json.dump(ground_truth, f, indent=2)

            if ground_truth.get("has_experiments"):
                print(f"  Found: {len(ground_truth['baselines'])} baselines, {len(ground_truth['metrics'])} metrics, {len(ground_truth['datasets'])} datasets")

                # Score alignments
                print("\n[Stage 3a] Non-agentic vs Ground Truth...")
                non_agentic_alignment = align_scorer.score_alignment(
                    non_agentic_result,
                    ground_truth
                )

                print(f"  Alignment: {non_agentic_alignment['overall_alignment']:.1%}")

                print("\n[Stage 3b] Agentic vs Ground Truth...")
                agentic_alignment = align_scorer.score_alignment(
                    agentic_result,
                    ground_truth
                )

                print(f"  Alignment: {agentic_alignment['overall_alignment']:.1%}")

                # Save alignments
                with open(output_dir / "non_agentic_alignment.json", 'w') as f:
                    json.dump(non_agentic_alignment, f, indent=2)

                with open(output_dir / "agentic_alignment.json", 'w') as f:
                    json.dump(agentic_alignment, f, indent=2)

                # Generate report
                report = format_alignment_report(non_agentic_alignment, "Non-Agentic")
                report += "\n" + format_alignment_report(agentic_alignment, "Agentic")

                with open(output_dir / "alignment_report.txt", 'w') as f:
                    f.write(report)

                # Determine winner
                if non_agentic_alignment['overall_alignment'] > agentic_alignment['overall_alignment']:
                    winner = "non_agentic"
                elif agentic_alignment['overall_alignment'] > non_agentic_alignment['overall_alignment']:
                    winner = "agentic"
                else:
                    winner = "tie"

                print(f"\n  Winner vs GT: {winner.upper()}")
            else:
                print("  No experiments found in paper")
                non_agentic_alignment = {"overall_alignment": 0, "confidence": "none"}
                agentic_alignment = {"overall_alignment": 0, "confidence": "none"}
                winner = "n/a"

            # Summary
            result_summary = {
                "paper_id": paper_id,
                "quality": {
                    "non_agentic": non_agentic_score["overall_score"],
                    "agentic": agentic_result["metadata"]["best_score"]
                },
                "alignment": {
                    "non_agentic": non_agentic_alignment["overall_alignment"],
                    "agentic": agentic_alignment["overall_alignment"],
                    "winner": winner
                },
                "cost": {
                    "non_agentic_time": non_agentic_result["metadata"]["time_elapsed"],
                    "agentic_time": agentic_result["metadata"]["time_elapsed"],
                    "non_agentic_calls": non_agentic_result["metadata"]["llm_calls"],
                    "agentic_calls": agentic_result["metadata"]["llm_calls"]
                }
            }

            results.append(result_summary)

        except Exception as e:
            print(f"\nERROR processing {paper_id}: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                "paper_id": paper_id,
                "error": str(e)
            })

    # Save overall summary
    summary = {
        "timestamp": timestamp,
        "total_papers": len(papers),
        "successful": len([r for r in results if "error" not in r]),
        "results": results
    }

    with open(run_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Results: {run_dir}")
    print(f"Successful: {summary['successful']}/{summary['total_papers']}")

    if summary['successful'] > 0:
        print("\nAlignment Winners:")
        for r in results:
            if "alignment" in r:
                print(f"  {r['paper_id'][:40]}: {r['alignment']['winner']}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
