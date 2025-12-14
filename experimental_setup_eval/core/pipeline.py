"""
Unified Evaluation Pipeline Logic.

Contains the core SimpleEvaluationPipeline class, separating logic from CLI orchestration.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List

from experimental_setup_eval.evaluation.data_models import (
    PaperContext, GroundTruth, extract_setup_from_dict
)
from experimental_setup_eval.evaluation.unified_evaluator import UnifiedEvaluator
from experimental_setup_eval.tools import PaperProcessorTool, GroundTruthExtractor
from experimental_setup_eval.generators.non_agentic_generator import NonAgenticSetupGenerator
from experimental_setup_eval.generators.agentic_generator import AgenticSetupGenerator


class SimpleEvaluationPipeline:
    """
    Simplified pipeline using unified components without circular imports.
    """

    def __init__(self, config, model: str):
        """Initialize pipeline with configuration and model."""
        self.config = config
        self.model = model

        # Initialize components
        self.processor = PaperProcessorTool(model=model)
        self.extractor = GroundTruthExtractor(model=model)
        self.evaluator = UnifiedEvaluator(config.evaluation)

        # Initialize generators
        self.non_agentic_gen = NonAgenticSetupGenerator(model=model)
        self.agentic_gen = AgenticSetupGenerator(
            model=model,
            num_drafts=config.evaluation.agentic.num_drafts,
            evaluator=self.evaluator
        )

    def run_single_paper(self, paper_path: Path, output_dir: Path = None) -> Dict[str, Any]:
        """Process a single paper through the evaluation pipeline."""
        paper_id = paper_path.parent.name
        start_time = time.time()

        try:
            # Stage 1: Process paper
            print("[Stage 1] Processing paper...")
            paper_result = self.processor.use_tool(
                str(paper_path),
                redact_sections=self.config.pipeline.redact_sections
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

            if output_dir:
                with open(output_dir / "paper_context.json", 'w') as f:
                    json.dump(paper_context.__dict__, f, indent=2, default=str)

            # Stage 2: Generate setups
            print("\n[Stage 2a] Non-agentic generation...")
            non_agentic_result = self.non_agentic_gen.generate_setup(
                paper_context.__dict__, paper_id
            )

            # Score non-agentic setup
            non_agentic_setup = extract_setup_from_dict(non_agentic_result["setup"])
            non_agentic_quality = self.evaluator.score_quality(
                non_agentic_setup, paper_context
            )

            print(f"  Quality: {non_agentic_quality.overall_score:.3f}")

            print("\n[Stage 2b] Agentic generation...")
            agentic_result = self.agentic_gen.generate_setup(
                paper_context.__dict__, paper_id
            )

            # Score agentic setup
            agentic_setup = extract_setup_from_dict(agentic_result["setup"])
            agentic_quality = self.evaluator.score_quality(
                agentic_setup, paper_context
            )

            print(f"  Quality: {agentic_quality.overall_score:.3f}")

            # Stage 3: Ground truth extraction
            print("\n[Stage 3] Extracting ground truth...")
            with open(paper_path) as f:
                paper_markdown = f.read()

            ground_truth_raw = self.extractor.extract_from_paper(paper_markdown)
            ground_truth = GroundTruth(
                baselines=ground_truth_raw.get("baselines", []),
                metrics=ground_truth_raw.get("metrics", []),
                datasets=ground_truth_raw.get("datasets", []),
                has_experiments=ground_truth_raw.get("has_experiments", False)
            )

            # Stage 4: Alignment scoring (if ground truth available)
            non_agentic_alignment = None
            agentic_alignment = None
            winner = "n/a"

            if ground_truth.has_experiments:
                print(f"  Found: {len(ground_truth.baselines)} baselines, {len(ground_truth.metrics)} metrics")

                non_agentic_alignment = self.evaluator.score_alignment(
                    non_agentic_setup, ground_truth
                )
                agentic_alignment = self.evaluator.score_alignment(
                    agentic_setup, ground_truth
                )

                print(f"  Non-agentic alignment: {non_agentic_alignment.overall_alignment:.1%}")
                print(f"  Agentic alignment: {agentic_alignment.overall_alignment:.1%}")

                # Determine winner
                if non_agentic_alignment.overall_alignment > agentic_alignment.overall_alignment:
                    winner = "non_agentic"
                elif agentic_alignment.overall_alignment > non_agentic_alignment.overall_alignment:
                    winner = "agentic"
                else:
                    winner = "tie"

                print(f"  Winner: {winner.upper()}")
            else:
                print("  No experiments found in paper")

            # Save results
            if output_dir:
                results = {
                    "paper_id": paper_id,
                    "non_agentic": {
                        **non_agentic_result,
                        "quality": non_agentic_quality.__dict__,
                        "alignment": non_agentic_alignment.__dict__ if non_agentic_alignment else None
                    },
                    "agentic": {
                        **agentic_result,
                        "quality": agentic_quality.__dict__,
                        "alignment": agentic_alignment.__dict__ if agentic_alignment else None
                    },
                    "ground_truth": ground_truth.__dict__,
                    "winner": winner,
                    "processing_time": time.time() - start_time
                }

                with open(output_dir / "results.json", 'w') as f:
                    json.dump(results, f, indent=2, default=str)

            return {
                "paper_id": paper_id,
                "success": True,
                "quality": {
                    "non_agentic": non_agentic_quality.overall_score,
                    "agentic": agentic_quality.overall_score
                },
                "alignment": {
                    "non_agentic": non_agentic_alignment.overall_alignment if non_agentic_alignment else 0,
                    "agentic": agentic_alignment.overall_alignment if agentic_alignment else 0,
                    "winner": winner
                },
                "has_experiments": ground_truth.has_experiments,
                "processing_time": time.time() - start_time
            }

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

            return {
                "paper_id": paper_id,
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }


def generate_summary(results: List[Dict], timestamp: str, run_dir: str) -> Dict:
    """Generate run summary with statistics."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    summary = {
        "timestamp": timestamp,
        "run_dir": run_dir,
        "total_papers": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(results) if results else 0.0
    }

    if successful:
        # Quality statistics
        non_agentic_scores = [r["quality"]["non_agentic"] for r in successful]
        agentic_scores = [r["quality"]["agentic"] for r in successful]

        summary["quality_stats"] = {
            "non_agentic": {
                "mean": sum(non_agentic_scores) / len(non_agentic_scores),
                "max": max(non_agentic_scores),
                "min": min(non_agentic_scores)
            },
            "agentic": {
                "mean": sum(agentic_scores) / len(agentic_scores),
                "max": max(agentic_scores),
                "min": min(agentic_scores)
            }
        }

        # Alignment statistics (for papers with ground truth)
        with_gt = [r for r in successful if r["has_experiments"]]
        if with_gt:
            alignment_winners = {}
            for r in with_gt:
                winner = r["alignment"]["winner"]
                alignment_winners[winner] = alignment_winners.get(winner, 0) + 1

            summary["alignment_stats"] = {
                "papers_with_experiments": len(with_gt),
                "winners": alignment_winners
            }

        # Performance statistics
        total_time = sum(r["processing_time"] for r in successful)

        summary["performance"] = {
            "total_time_seconds": total_time,
            "avg_time_per_paper": total_time / len(successful)
        }

    # Error summary
    if failed:
        summary["errors"] = [
            {"paper_id": r["paper_id"], "error": r["error"]}
            for r in failed
        ]

    return summary


def print_final_summary(summary: Dict, results: List[Dict]) -> None:
    """Print final summary to console."""
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")

    print(f"Total papers: {summary['total_papers']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success rate: {summary['success_rate']:.1%}")

    if "quality_stats" in summary:
        print(f"\nQuality Scores:")
        non_ag = summary["quality_stats"]["non_agentic"]
        agentic = summary["quality_stats"]["agentic"]
        print(f"  Non-agentic: {non_ag['mean']:.3f} ± {non_ag['max'] - non_ag['min']:.3f}")
        print(f"  Agentic: {agentic['mean']:.3f} ± {agentic['max'] - agentic['min']:.3f}")

        improvement = agentic["mean"] - non_ag["mean"]
        print(f"  Improvement: {improvement:+.3f} ({improvement/non_ag['mean']*100:+.1f}%)")

    if "alignment_stats" in summary:
        print(f"\nAlignment Results ({summary['alignment_stats']['papers_with_experiments']} papers with experiments):")
        for winner, count in summary["alignment_stats"]["winners"].items():
            print(f"  {winner}: {count} papers")

    if "performance" in summary:
        perf = summary["performance"]
        print(f"\nPerformance:")
        print(f"  Total time: {perf['total_time_seconds']:.1f}s")
        print(f"  Avg per paper: {perf['avg_time_per_paper']:.1f}s")

    import os
    try:
        rel_run_dir = os.path.relpath(summary['run_dir'])
    except ValueError:
        rel_run_dir = summary['run_dir']

    print(f"\nResults saved to: {rel_run_dir}")
    print(f"{'='*70}\n")
