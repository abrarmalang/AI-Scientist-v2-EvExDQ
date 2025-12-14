"""
Clean Pipeline Orchestrator - Replaces complex run_pipeline.py logic.
Handles experimental setup evaluation with dependency injection and clean error handling.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

# Handle imports for both package and standalone usage
try:
    from .unified_evaluator import UnifiedEvaluator
    from .data_models import (
        PipelineResult, PaperContext, ExperimentalSetup, GroundTruth,
        SetupGenerationResult, GenerationMetadata, ConfidenceLevel,
        extract_setup_from_dict
    )
    from ..core.schema import EvalSetupConfig
    from ..tools import PaperProcessorTool, GroundTruthExtractor
    from ..generators import NonAgenticSetupGenerator, AgenticSetupGenerator
except ImportError:
    # Fallback for standalone usage - add directories to path
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    if str(parent_dir.parent) not in sys.path:  # for ai_scientist
        sys.path.insert(0, str(parent_dir.parent))

    from evaluation.unified_evaluator import UnifiedEvaluator
    from evaluation.data_models import (
        PipelineResult, PaperContext, ExperimentalSetup, GroundTruth,
        SetupGenerationResult, GenerationMetadata, ConfidenceLevel,
        extract_setup_from_dict
    )
    from experimental_setup_eval.core.schema import EvalSetupConfig
    from tools import PaperProcessorTool, GroundTruthExtractor
    from generators import NonAgenticSetupGenerator, AgenticSetupGenerator


class EvaluationPipeline:
    """
    Clean pipeline orchestrator with dependency injection.
    Replaces complex 280-line main function with organized, testable components.
    """

    def __init__(self, config: EvalSetupConfig, model: str):
        """
        Initialize pipeline with configuration and model.

        Args:
            config: Validated configuration
            model: Model name to use for all components
        """
        self.config = config
        self.model = model

        # Initialize components with dependency injection
        self.processor = PaperProcessorTool(model=model)
        self.extractor = GroundTruthExtractor(model=model)
        self.evaluator = UnifiedEvaluator(config.evaluation)

        # Initialize generators
        self.non_agentic_gen = NonAgenticSetupGenerator(model=model)
        self.agentic_gen = AgenticSetupGenerator(
            model=model,
            num_drafts=config.evaluation.agentic.num_drafts
        )

    def run_single_paper(
        self,
        paper_path: Path,
        output_dir: Optional[Path] = None
    ) -> PipelineResult:
        """
        Process a single paper through the complete evaluation pipeline.

        Args:
            paper_path: Path to paper markdown file
            output_dir: Optional output directory for intermediate results

        Returns:
            Complete pipeline result for the paper
        """
        paper_id = paper_path.parent.name
        start_time = time.time()

        try:
            # Stage 1: Process paper
            paper_context = self._process_paper(paper_path)

            if output_dir:
                self._save_json(paper_context.__dict__, output_dir / "paper_context.json")

            # Stage 2: Generate setups
            non_agentic_result = self._generate_non_agentic_setup(paper_context, paper_id)
            agentic_result = self._generate_agentic_setup(paper_context, paper_id)

            if output_dir:
                self._save_json(non_agentic_result.__dict__, output_dir / "non_agentic_setup.json")
                self._save_json(agentic_result.__dict__, output_dir / "agentic_setup.json")

            # Stage 3: Ground truth extraction and alignment
            ground_truth = self._extract_ground_truth(paper_path)

            if output_dir:
                self._save_json(ground_truth.__dict__, output_dir / "ground_truth.json")

            # Stage 4: Comparative evaluation
            comparison_result = self._compare_setups(non_agentic_result, agentic_result, ground_truth)

            if output_dir:
                self._save_json(comparison_result.__dict__, output_dir / "comparison.json")

            # Create complete result
            return PipelineResult(
                paper_id=paper_id,
                paper_context=paper_context,
                non_agentic_result=non_agentic_result,
                agentic_result=agentic_result,
                ground_truth=ground_truth,
                comparison_result=comparison_result,
                success=True
            )

        except Exception as e:
            return PipelineResult(
                paper_id=paper_id,
                paper_context=PaperContext("", "", "", "", "", "", ""),
                non_agentic_result=self._empty_generation_result(paper_id, "non_agentic"),
                agentic_result=self._empty_generation_result(paper_id, "agentic"),
                ground_truth=GroundTruth(),
                comparison_result=self.evaluator.compare_setups({}, {}, ("failed", "failed")),
                success=False,
                error=str(e)
            )

    def run_batch(
        self,
        paper_paths: List[Path],
        run_dir: Path
    ) -> List[PipelineResult]:
        """
        Process multiple papers in batch.

        Args:
            paper_paths: List of paper paths to process
            run_dir: Base directory for run outputs

        Returns:
            List of pipeline results
        """
        results = []

        for i, paper_path in enumerate(paper_paths, 1):
            paper_id = paper_path.parent.name

            print(f"\n{'='*70}")
            print(f"[{i}/{len(paper_paths)}] {paper_id[:50]}")
            print(f"{'='*70}\n")

            # Create paper-specific output directory
            output_dir = run_dir / paper_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Process paper
            result = self.run_single_paper(paper_path, output_dir)
            results.append(result)

            # Print summary
            if result.success:
                print(f"✓ Success")
                print(f"  Non-agentic quality: {result.non_agentic_result.quality_result.overall_score:.3f}")
                print(f"  Agentic quality: {result.agentic_result.quality_result.overall_score:.3f}")
                if result.ground_truth.has_experiments:
                    print(f"  Alignment winner: {result.comparison_result.winner}")
            else:
                print(f"✗ Failed: {result.error}")

        return results

    # Private helper methods

    def _process_paper(self, paper_path: Path) -> PaperContext:
        """Process paper and extract context."""
        paper_result = self.processor.use_tool(
            str(paper_path),
            redact_sections=self.config.pipeline.redact_sections
        )

        return PaperContext(
            title=paper_result["title"],
            abstract=paper_result["abstract"],
            introduction=paper_result["introduction"],
            related_work=paper_result["related_work"],
            research_question=paper_result["research_question"],
            domain=paper_result["domain"],
            method_description=paper_result["method_description"]
        )

    def _generate_non_agentic_setup(
        self,
        context: PaperContext,
        paper_id: str
    ) -> SetupGenerationResult:
        """Generate non-agentic setup with quality scoring."""
        start_time = time.time()

        # Generate setup
        raw_result = self.non_agentic_gen.generate_setup(context.__dict__, paper_id)

        # Extract setup and metadata
        setup = extract_setup_from_dict(raw_result["setup"])
        metadata = GenerationMetadata(**raw_result["metadata"])

        # Score quality
        quality_result = self.evaluator.score_quality(setup, context)
        metadata.quality_score = quality_result.overall_score

        return SetupGenerationResult(
            setup=setup,
            metadata=metadata,
            quality_result=quality_result
        )

    def _generate_agentic_setup(
        self,
        context: PaperContext,
        paper_id: str
    ) -> SetupGenerationResult:
        """Generate agentic setup with quality scoring."""
        # Generate setup
        raw_result = self.agentic_gen.generate_setup(context.__dict__, paper_id)

        # Extract best setup and metadata
        if "ai_setup" in raw_result:
            setup = extract_setup_from_dict(raw_result)
        else:
            setup = extract_setup_from_dict(raw_result["setup"])

        metadata = GenerationMetadata(**raw_result["metadata"])

        # Score quality
        quality_result = self.evaluator.score_quality(setup, context)

        return SetupGenerationResult(
            setup=setup,
            metadata=metadata,
            quality_result=quality_result
        )

    def _extract_ground_truth(self, paper_path: Path) -> GroundTruth:
        """Extract ground truth from paper."""
        # Read full paper for extraction
        with open(paper_path) as f:
            paper_markdown = f.read()

        raw_gt = self.extractor.extract_from_paper(paper_markdown)

        return GroundTruth(
            baselines=raw_gt.get("baselines", []),
            metrics=raw_gt.get("metrics", []),
            datasets=raw_gt.get("datasets", []),
            significance_tests=raw_gt.get("significance_tests", []),
            experimental_details=raw_gt.get("experimental_details", {}),
            has_experiments=raw_gt.get("has_experiments", False),
            extraction_confidence=ConfidenceLevel.MEDIUM if raw_gt.get("has_experiments") else ConfidenceLevel.NONE
        )

    def _compare_setups(
        self,
        non_agentic: SetupGenerationResult,
        agentic: SetupGenerationResult,
        ground_truth: GroundTruth
    ) -> Any:
        """Compare setups and score alignments."""
        # Score alignments if ground truth is available
        if ground_truth.has_experiments:
            non_agentic.alignment_result = self.evaluator.score_alignment(
                non_agentic.setup, ground_truth
            )
            agentic.alignment_result = self.evaluator.score_alignment(
                agentic.setup, ground_truth
            )

        # Compare setups
        return self.evaluator.compare_setups(
            {
                "setup": non_agentic.setup.__dict__,
                "metadata": non_agentic.metadata.__dict__
            },
            {
                "setup": agentic.setup.__dict__,
                "metadata": agentic.metadata.__dict__
            },
            labels=("non_agentic", "agentic"),
            ground_truth=ground_truth.__dict__ if ground_truth.has_experiments else None
        )

    def _empty_generation_result(self, paper_id: str, approach: str) -> SetupGenerationResult:
        """Create empty generation result for error cases."""
        return SetupGenerationResult(
            setup=ExperimentalSetup(),
            metadata=GenerationMetadata(
                paper_id=paper_id,
                timestamp=datetime.now().isoformat(),
                model=self.model,
                approach=approach,
                time_elapsed=0.0,
                llm_calls=0,
                description="Failed"
            )
        )

    def _save_json(self, data: Any, path: Path) -> None:
        """Save data as JSON with error handling."""
        if self.config.save_intermediate:
            try:
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception as e:
                print(f"Warning: Failed to save {path}: {e}")


def create_pipeline(config: EvalSetupConfig, model: str) -> EvaluationPipeline:
    """Factory function to create configured pipeline."""
    return EvaluationPipeline(config, model)