"""
Unified Evaluator - Replaces multiple separate evaluator classes.
Provides quality scoring, alignment scoring, and setup comparison in a single clean interface.
"""

from typing import Dict, List, Any, Set, Tuple, Optional
import re
import json
from collections import defaultdict

# Handle imports for both package and standalone usage
try:
    from .data_models import (
        EvaluationResult, QualityResult, AlignmentResult, ComparisonResult,
        ExperimentalSetup, PaperContext, GroundTruth, ComponentAlignment,
        ConfidenceLevel, extract_setup_from_dict, calculate_component_alignment
    )
    from ..core.schema import EvaluationConfig
    from ..core.prompts import create_alignment_prompt
    from ..backends.native_anthropic_backend import smart_query
except ImportError:
    # Fallback for standalone usage - add current directory to path
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    from evaluation.data_models import (
        EvaluationResult, QualityResult, AlignmentResult, ComparisonResult,
        ExperimentalSetup, PaperContext, GroundTruth, ComponentAlignment,
        ConfidenceLevel, extract_setup_from_dict, calculate_component_alignment
    )
    from core.schema import EvaluationConfig
    from core.prompts import create_alignment_prompt
    from backends.native_anthropic_backend import smart_query


class UnifiedEvaluator:
    """
    Unified evaluator combining quality scoring, alignment, and comparison.
    Replaces SetupQualityScorer, AlignmentScorer, SemanticAlignmentScorer, and SetupComparator.
    """

    def __init__(self, config: EvaluationConfig):
        """Initialize with evaluation configuration."""
        self.config = config
        self.quality_weights = config.quality_weights
        self.alignment_weights = config.alignment_weights

    def score_quality(
        self,
        setup_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> QualityResult:
        """
        Score experimental setup quality without ground truth.

        Args:
            setup_data: Setup dictionary or ExperimentalSetup
            context: Paper context for domain alignment

        Returns:
            QualityResult with component scores and feedback
        """
        # Normalize input
        if isinstance(setup_data, dict):
            setup = extract_setup_from_dict(setup_data)
        else:
            setup = setup_data

        paper_context = PaperContext(**context) if isinstance(context, dict) else context

        # Calculate component scores
        completeness = self._score_completeness(setup)
        diversity = self._score_baseline_diversity(setup)
        alignment = self._score_metric_alignment(setup, paper_context)
        datasets = self._score_dataset_appropriateness(setup, paper_context)
        protocol = self._score_protocol_rigor(setup)
        
        # New rigorous scores
        hypotheses = self._score_hypotheses_clarity(setup)
        rigor = self._score_methodological_rigor(setup)
        controls = self._score_control_adequacy(setup)
        validity = self._score_validity_awareness(setup)

        # Calculate weighted overall score
        # Adjust weights to include new components (renormalizing to sum to ~1.0)
        # Old weights: completeness, diversity, alignment, datasets, protocol
        # New mix: 
        # - Core (50%): completeness, alignment, protocol, rigor
        # - Content (30%): diversity, datasets, controls
        # - Scientific (20%): hypotheses, validity
        
        overall_score = (
            0.15 * completeness +
            0.10 * diversity +
            0.15 * alignment +
            0.10 * datasets +
            0.15 * protocol +
            0.10 * hypotheses +
            0.10 * rigor +
            0.10 * controls +
            0.05 * validity
        )

        # Generate feedback
        feedback = self._generate_quality_feedback(
            completeness, diversity, alignment, datasets, protocol, overall_score
        )
        
        # Add feedback for new components
        if hypotheses < 0.5:
            feedback["weaknesses"].append("Hypotheses lack clarity or testability")
            feedback["suggestions"].append("Formulate clear, testable hypotheses")
        if rigor < 0.5:
            feedback["weaknesses"].append("Methodological rigor is insufficient")
            feedback["suggestions"].append("Justify sample size and detail randomization/blinding")
        if controls < 0.5:
            feedback["weaknesses"].append("Control groups are inadequate")
            feedback["suggestions"].append("Include specific control conditions with rationale")

        # Determine confidence based on completeness and overall score
        confidence = self._calculate_confidence(completeness, overall_score)

        return QualityResult(
            overall_score=overall_score,
            component_scores={
                "completeness": completeness,
                "diversity": diversity,
                "alignment": alignment,
                "datasets": datasets,
                "protocol": protocol,
                "hypotheses": hypotheses,
                "rigor": rigor,
                "controls": controls,
                "validity": validity
            },
            confidence=confidence,
            completeness=completeness,
            diversity=diversity,
            alignment=alignment,
            datasets=datasets,
            protocol=protocol,
            feedback=feedback
        )

    def score_alignment(
        self,
        setup_data: Dict[str, Any],
        ground_truth_data: Dict[str, Any]
    ) -> AlignmentResult:
        """
        Score alignment between generated setup and ground truth.
        
        Enhanced to use F1 scores, handle partial ground truth,
        and include qualitative alignment for hypotheses/controls/protocols.

        Args:
            setup_data: Generated setup
            ground_truth_data: Ground truth setup

        Returns:
            AlignmentResult with component-wise alignment scores
        """
        # Normalize inputs
        if isinstance(setup_data, dict):
            setup = extract_setup_from_dict(setup_data)
        else:
            setup = setup_data

        ground_truth = GroundTruth(**ground_truth_data) if isinstance(ground_truth_data, dict) else ground_truth_data

        if not ground_truth.has_experiments:
            return AlignmentResult(
                overall_score=0.0,
                component_scores={},
                confidence=ConfidenceLevel.NONE,
                overall_alignment=0.0
            )

        # Score each quantitative component with availability checking
        available_components = []
        baseline_alignment = None
        metric_alignment = None
        dataset_alignment = None
        
        if ground_truth.baselines:
            baseline_alignment = calculate_component_alignment(setup.baselines, ground_truth.baselines)
            available_components.append(('baseline', baseline_alignment.f1_score, self.alignment_weights.baseline_weight))
        
        if ground_truth.metrics:
            metric_alignment = calculate_component_alignment(setup.metrics, ground_truth.metrics)
            available_components.append(('metric', metric_alignment.f1_score, self.alignment_weights.metric_weight))
        
        if ground_truth.datasets:
            dataset_alignment = calculate_component_alignment(setup.datasets, ground_truth.datasets)
            available_components.append(('dataset', dataset_alignment.f1_score, self.alignment_weights.dataset_weight))
        
        # Score qualitative alignments using existing helper methods
        hypotheses_alignment = self._score_text_similarity(
            setup.hypotheses, ground_truth.experimental_details.get("hypotheses", [])
        )
        
        control_alignment = self._score_list_overlap(
            setup.control_groups, ground_truth.experimental_details.get("controls", [])
        )
        
        protocol_alignment = self._score_protocol_match(
            setup.experimental_protocol, ground_truth.experimental_details
        )

        # Calculate weighted overall alignment with normalization for available components
        if available_components:
            total_available_weight = sum(weight for _, _, weight in available_components)
            overall_alignment = sum(
                score * (weight / total_available_weight) 
                for _, score, weight in available_components
            )
        else:
            overall_alignment = 0.0
        
        # Build component scores dictionary
        component_scores = {}
        if baseline_alignment:
            component_scores.update({
                "baseline_recall": baseline_alignment.recall,
                "baseline_precision": baseline_alignment.precision,
                "baseline_f1": baseline_alignment.f1_score
            })
        if metric_alignment:
            component_scores.update({
                "metric_recall": metric_alignment.recall,
                "metric_precision": metric_alignment.precision,
                "metric_f1": metric_alignment.f1_score
            })
        if dataset_alignment:
            component_scores.update({
                "dataset_recall": dataset_alignment.recall,
                "dataset_precision": dataset_alignment.precision,
                "dataset_f1": dataset_alignment.f1_score
            })
        
        # Add qualitative scores
        component_scores.update({
            "hypotheses_alignment": hypotheses_alignment,
            "control_alignment": control_alignment,
            "protocol_alignment": protocol_alignment
        })
        
        # Determine confidence with enhanced calculation
        confidence = self._calculate_alignment_confidence_enhanced(
            baseline_alignment, metric_alignment, dataset_alignment,
            ground_truth, len(available_components)
        )

        return AlignmentResult(
            overall_score=overall_alignment,
            component_scores=component_scores,
            confidence=confidence,
            baseline_alignment=baseline_alignment or ComponentAlignment(0, 0, 0, [], [], []),
            metric_alignment=metric_alignment or ComponentAlignment(0, 0, 0, [], [], []),
            dataset_alignment=dataset_alignment or ComponentAlignment(0, 0, 0, [], [], []),
            hypotheses_alignment=hypotheses_alignment,
            control_alignment=control_alignment,
            protocol_alignment=protocol_alignment,
            overall_alignment=overall_alignment
        )

    def compare_setups(
        self,
        setup_a: Dict[str, Any],
        setup_b: Dict[str, Any],
        labels: Tuple[str, str] = ("Setup A", "Setup B"),
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> ComparisonResult:
        """
        Compare two experimental setups.

        Args:
            setup_a: First setup with metadata
            setup_b: Second setup with metadata
            labels: Labels for the setups
            ground_truth: Optional ground truth for alignment comparison

        Returns:
            ComparisonResult with winner determination and detailed comparison
        """
        label_a, label_b = labels

        # Extract setups and metadata
        actual_setup_a = extract_setup_from_dict(setup_a.get("setup", setup_a))
        actual_setup_b = extract_setup_from_dict(setup_b.get("setup", setup_b))
        metadata_a = setup_a.get("metadata", {})
        metadata_b = setup_b.get("metadata", {})

        # Quality comparison
        quality_a = metadata_a.get("quality_score", 0.0)
        quality_b = metadata_b.get("quality_score", 0.0)

        # Cost comparison
        cost_comparison = {
            "time_ratio": metadata_b.get("time_elapsed", 1) / max(metadata_a.get("time_elapsed", 1), 0.001),
            "calls_ratio": metadata_b.get("llm_calls", 1) / max(metadata_a.get("llm_calls", 1), 0.001)
        }

        # Component overlap (Jaccard similarity)
        component_overlap = {
            "baselines": self._jaccard_similarity(actual_setup_a.baselines, actual_setup_b.baselines),
            "metrics": self._jaccard_similarity(actual_setup_a.metrics, actual_setup_b.metrics),
            "datasets": self._jaccard_similarity(actual_setup_a.datasets, actual_setup_b.datasets)
        }

        # Determine winner
        if abs(quality_a - quality_b) < 0.01:  # Very close scores
            winner = "tie"
        elif quality_a > quality_b:
            winner = label_a
        else:
            winner = label_b

        # Generate summary
        summary = self._generate_comparison_summary(
            quality_a, quality_b, cost_comparison, component_overlap, labels, winner
        )

        return ComparisonResult(
            winner=winner,
            quality_comparison={label_a: quality_a, label_b: quality_b},
            cost_comparison=cost_comparison,
            component_overlap=component_overlap,
            summary=summary,
            details={
                "metadata_a": metadata_a,
                "metadata_b": metadata_b,
                "quality_difference": abs(quality_a - quality_b)
            }
        )

    # Private helper methods

    def _score_completeness(self, setup: ExperimentalSetup) -> float:
        """Score setup completeness (0-1)."""
        components = {
            "baselines": len(setup.baselines) > 0,
            "metrics": len(setup.metrics) > 0,
            "datasets": len(setup.datasets) > 0,
            "protocol": bool(setup.experimental_protocol),
            "reasoning": bool(setup.reasoning.strip())
        }

        score = sum(components.values()) / len(components)

        # Bonus for multiple items in each component
        if len(setup.baselines) >= 3:
            score += 0.1
        if len(setup.metrics) >= 2:
            score += 0.1

        return min(score, 1.0)

    def _score_baseline_diversity(self, setup: ExperimentalSetup) -> float:
        """Score baseline diversity (0-1)."""
        if not setup.baselines:
            return 0.0

        # Check for different baseline types
        baseline_types = {
            "random": any("random" in b.lower() for b in setup.baselines),
            "simple": any(any(term in b.lower() for term in ["simple", "naive", "basic"]) for b in setup.baselines),
            "sota": any(any(term in b.lower() for term in ["bert", "gpt", "transformer", "neural"]) for b in setup.baselines),
            "traditional": any(any(term in b.lower() for term in ["svm", "knn", "linear", "tree"]) for b in setup.baselines)
        }

        diversity_score = sum(baseline_types.values()) / len(baseline_types)

        # Bonus for having multiple baselines
        count_bonus = min(len(setup.baselines) * 0.1, 0.3)

        return min(diversity_score + count_bonus, 1.0)

    def _score_metric_alignment(self, setup: ExperimentalSetup, context: PaperContext) -> float:
        """Score metric alignment with paper domain (0-1)."""
        if not setup.metrics:
            return 0.0

        domain_metrics = {
            "nlp": ["bleu", "rouge", "perplexity", "f1", "accuracy", "precision", "recall"],
            "cv": ["map", "iou", "accuracy", "precision", "recall", "top-k"],
            "ml": ["accuracy", "f1", "auc", "mse", "mae", "r2"],
            "localization": ["error", "accuracy", "distance", "rmse"],
            "retrieval": ["map", "recall", "precision", "ndcg"]
        }

        domain = context.domain.lower()
        relevant_metrics = []

        for key, metrics in domain_metrics.items():
            if key in domain or any(key_term in context.title.lower() + context.abstract.lower()
                                  for key_term in [key]):
                relevant_metrics.extend(metrics)

        if not relevant_metrics:
            relevant_metrics = domain_metrics["ml"]  # fallback

        # Check alignment
        metric_names = [m.lower() for m in setup.metrics]
        aligned = sum(1 for m in metric_names if any(rm in m for rm in relevant_metrics))

        return min(aligned / len(setup.metrics), 1.0)

    def _score_dataset_appropriateness(self, setup: ExperimentalSetup, context: PaperContext) -> float:
        """Score dataset appropriateness (0-1)."""
        if not setup.datasets:
            return 0.0

        # Simple heuristic: datasets mentioned in context get higher scores
        context_text = (context.title + " " + context.abstract + " " +
                       context.introduction + " " + context.method_description).lower()

        appropriate = sum(1 for dataset in setup.datasets
                        if dataset.lower() in context_text or
                           any(word in context_text for word in dataset.lower().split()))

        score = appropriate / len(setup.datasets)

        # Bonus for having multiple datasets
        if len(setup.datasets) >= 2:
            score += 0.1

        return min(score, 1.0)

    def _score_protocol_rigor(self, setup: ExperimentalSetup) -> float:
        """Score experimental protocol rigor (0-1)."""
        if not setup.experimental_protocol:
            return 0.0

        protocol = setup.experimental_protocol
        rigor_checks = {
            "cross_validation": any(key in protocol for key in ["cv", "cross_validation", "folds"]),
            "multiple_runs": any(key in protocol for key in ["runs", "seeds", "trials"]),
            "significance_tests": len(setup.significance_tests) > 0,
            "train_test_split": any(key in protocol for key in ["split", "train", "test", "validation"])
        }

        return sum(rigor_checks.values()) / len(rigor_checks)

    def _generate_quality_feedback(
        self, completeness: float, diversity: float, alignment: float,
        datasets: float, protocol: float, overall: float
    ) -> Dict[str, List[str]]:
        """Generate quality feedback."""
        strengths = []
        weaknesses = []
        suggestions = []

        # Analyze each component
        if completeness >= 0.8:
            strengths.append("Setup has all required components")
        elif completeness < 0.5:
            weaknesses.append("Setup is missing key components")
            suggestions.append("Ensure all components (baselines, metrics, datasets, protocol) are specified")

        if diversity >= 0.7:
            strengths.append("Good baseline diversity")
        elif diversity < 0.5:
            weaknesses.append("Limited baseline diversity")
            suggestions.append("Include baselines of different complexity levels (simple, traditional, SOTA)")

        if alignment >= 0.8:
            strengths.append("Metrics align well with domain")
        elif alignment < 0.5:
            weaknesses.append("Metrics may not be appropriate for domain")
            suggestions.append("Choose metrics that are standard in the research domain")

        if protocol >= 0.7:
            strengths.append("Rigorous experimental protocol")
        elif protocol < 0.5:
            weaknesses.append("Experimental protocol lacks rigor")
            suggestions.append("Include cross-validation, multiple runs, and significance testing")

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "suggestions": suggestions
        }

    def _score_hypotheses_clarity(self, setup: ExperimentalSetup) -> float:
        """Score hypotheses clarity (0-1)."""
        if not setup.hypotheses:
            return 0.0
            
        # Check for testable phrasing
        testable_terms = ["significant", "increase", "decrease", "correlate", "affect", "impact", "than"]
        testable_count = sum(1 for h in setup.hypotheses if any(term in h.lower() for term in testable_terms))
        
        return min(testable_count / len(setup.hypotheses), 1.0)

    def _score_methodological_rigor(self, setup: ExperimentalSetup) -> float:
        """Score methodological rigor (0-1)."""
        score = 0.0
        
        # Sample size justification
        if setup.sample_size_analysis and len(setup.sample_size_analysis) > 20:
            score += 0.3
        if "power analysis" in setup.sample_size_analysis.lower():
            score += 0.2
            
        # Randomization and blinding
        if setup.randomization_blinding and len(setup.randomization_blinding) > 20:
            score += 0.3
        if "random" in setup.randomization_blinding.lower():
            score += 0.1
        if "blind" in setup.randomization_blinding.lower():
            score += 0.1
            
        return min(score, 1.0)

    def _score_control_adequacy(self, setup: ExperimentalSetup) -> float:
        """Score control group adequacy (0-1)."""
        if not setup.control_groups:
            return 0.0
            
        # Check for detailed descriptions
        detailed_controls = sum(1 for c in setup.control_groups if isinstance(c, dict) and "rationale" in c)
        
        # Bonus for multiple controls
        count_score = min(len(setup.control_groups) * 0.2, 0.6)
        detail_score = min(detailed_controls * 0.2, 0.4)
        
        return min(count_score + detail_score, 1.0)

    def _score_validity_awareness(self, setup: ExperimentalSetup) -> float:
        """Score validity awareness (0-1)."""
        if not setup.validity_assessments:
            return 0.0
            
        # Handle case where LLM returns string instead of dict
        if isinstance(setup.validity_assessments, str):
            # Simple keyword matching fallback
            content = setup.validity_assessments.lower()
            validity_types = ["internal validity", "external validity", "construct validity"]
            matches = sum(1 for v in validity_types if v in content)
            return min(matches * 0.33, 1.0)
            
        if not isinstance(setup.validity_assessments, dict):
            return 0.0
            
        validity_types = ["internal_validity", "external_validity", "construct_validity"]
        score = 0.0
        
        for v_type in validity_types:
            if setup.validity_assessments.get(v_type):
                score += 0.33
                
        return min(score, 1.0)

    def _calculate_confidence(self, completeness: float, overall: float) -> ConfidenceLevel:
        """Calculate confidence level."""
        if completeness >= 0.8 and overall >= 0.7:
            return ConfidenceLevel.HIGH
        elif completeness >= 0.6 and overall >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _calculate_alignment_confidence(
        self, baseline_align: ComponentAlignment, metric_align: ComponentAlignment,
        dataset_align: ComponentAlignment
    ) -> ConfidenceLevel:
        """Calculate alignment confidence (legacy method for backward compatibility)."""
        avg_recall = (baseline_align.recall + metric_align.recall + dataset_align.recall) / 3

        if avg_recall >= 0.7:
            return ConfidenceLevel.HIGH
        elif avg_recall >= 0.4:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _calculate_alignment_confidence_enhanced(
        self,
        baseline_align: Optional[ComponentAlignment],
        metric_align: Optional[ComponentAlignment],
        dataset_align: Optional[ComponentAlignment],
        ground_truth: GroundTruth,
        num_available_components: int
    ) -> ConfidenceLevel:
        """
        Enhanced alignment confidence calculation.
        
        Considers F1 scores, ground truth completeness, and extraction confidence.
        """
        # Collect F1 scores from available components
        f1_scores = []
        if baseline_align:
            f1_scores.append(baseline_align.f1_score)
        if metric_align:
            f1_scores.append(metric_align.f1_score)
        if dataset_align:
            f1_scores.append(dataset_align.f1_score)
        
        if not f1_scores:
            return ConfidenceLevel.NONE
        
        avg_f1 = sum(f1_scores) / len(f1_scores)
        
        # Factor in ground truth completeness (how many components are available)
        completeness_factor = num_available_components / 3.0  # 3 main components expected
        
        # Factor in ground truth extraction confidence
        extraction_confidence_map = {
            ConfidenceLevel.HIGH: 1.0,
            ConfidenceLevel.MEDIUM: 0.7,
            ConfidenceLevel.LOW: 0.5,
            ConfidenceLevel.NONE: 0.3
        }
        extraction_factor = extraction_confidence_map.get(ground_truth.extraction_confidence, 0.5)
        
        # Combined confidence score
        combined_score = avg_f1 * 0.6 + completeness_factor * 0.2 + extraction_factor * 0.2
        
        # Determine confidence level
        if combined_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif combined_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _jaccard_similarity(self, list_a: List[str], list_b: List[str]) -> Dict[str, float]:
        """Calculate Jaccard similarity between two component lists."""
        if not list_a and not list_b:
            return {"similarity": 1.0, "intersection": 0, "union": 0}

        set_a = set(ExperimentalSetup._normalize_name(item) for item in list_a)
        set_b = set(ExperimentalSetup._normalize_name(item) for item in list_b)

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        similarity = intersection / union if union > 0 else 0.0

        return {
            "similarity": similarity,
            "intersection": intersection,
            "union": union
        }

    def _generate_comparison_summary(
        self, quality_a: float, quality_b: float, cost_comparison: Dict[str, float],
        component_overlap: Dict[str, Dict[str, float]], labels: Tuple[str, str], winner: str
    ) -> str:
        """Generate comparison summary."""
        label_a, label_b = labels

        summary_parts = [f"Winner: {winner}"]

        quality_diff = abs(quality_a - quality_b)
        if quality_diff < 0.01:
            summary_parts.append("Quality scores are very close")
        else:
            better_label = label_a if quality_a > quality_b else label_b
            summary_parts.append(f"{better_label} has {quality_diff:.3f} higher quality score")

        # Cost analysis
        if cost_comparison["time_ratio"] > 2:
            summary_parts.append(f"{label_b} took {cost_comparison['time_ratio']:.1f}x longer")
        elif cost_comparison["time_ratio"] < 0.5:
            summary_parts.append(f"{label_b} was {1/cost_comparison['time_ratio']:.1f}x faster")

        # Component overlap
        avg_overlap = sum(comp["similarity"] for comp in component_overlap.values()) / len(component_overlap)
        if avg_overlap > 0.7:
            summary_parts.append("High component overlap between approaches")
        elif avg_overlap < 0.3:
            summary_parts.append("Low component overlap - approaches differ significantly")

        return ". ".join(summary_parts) + "."

    def _semantic_component_alignment(
        self,
        generated: List[str],
        ground_truth: List[str],
        component_name: str
    ) -> ComponentAlignment:
        """
        Use LLM to semantically compare component lists.
        
        Args:
            generated: Generated component list
            ground_truth: Ground truth component list
            component_name: Name of component for prompt
            
        Returns:
            ComponentAlignment with semantic matching
        """
        if not generated or not ground_truth:
            return ComponentAlignment(
                recall=0.0, precision=0.0, f1_score=0.0,
                found=[], missing=ground_truth, extra=generated
            )
        
        # Use LLM to evaluate semantic matches
        prompt = create_alignment_prompt(component_name, generated, ground_truth)
        
        try:
            response = smart_query(prompt, model="claude-3-haiku-20240307")
            result = json.loads(response)
            matches = result.get("matches", [])
            
            # Calculate recall based on coverage
            covered_count = sum(1 for m in matches if m.get("covered", False))
            recall = covered_count / len(ground_truth) if ground_truth else 0.0
            
            # Calculate precision (how many generated items matched something)
            matched_gen_items = set(m.get("generated_match") for m in matches if m.get("covered") and m.get("generated_match"))
            precision = len(matched_gen_items) / len(generated) if generated else 0.0
            
            # F1 score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Build found, missing, extra lists
            found = [m["ground_truth_item"] for m in matches if m.get("covered")]
            missing = [m["ground_truth_item"] for m in matches if not m.get("covered")]
            extra = [g for g in generated if g not in matched_gen_items]
            
            return ComponentAlignment(
                recall=recall,
                precision=precision,
                f1_score=f1_score,
                found=found,
                missing=missing,
                extra=extra
            )
        except Exception as e:
            # Fallback to exact matching if LLM fails
            print(f"Semantic alignment failed for {component_name}, using exact match: {e}")
            return calculate_component_alignment(generated, ground_truth)

    def _score_text_similarity(self, list_a: List[str], list_b: List[str]) -> float:
        """Score similarity between two text lists (0-1)."""
        if not list_a or not list_b:
            return 0.0
        
        # Simple overlap-based similarity
        set_a = set(s.lower().strip() for s in list_a)
        set_b = set(s.lower().strip() for s in list_b)
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0

    def _score_list_overlap(self, list_a: List[str], list_b: List[str]) -> float:
        """Score overlap between two lists (0-1)."""
        if not list_a or not list_b:
            return 0.0
        
        # Normalize and compare
        norm_a = set(ExperimentalSetup._normalize_name(item) for item in list_a)
        norm_b = set(ExperimentalSetup._normalize_name(item) for item in list_b)
        
        intersection = len(norm_a & norm_b)
        return intersection / len(norm_b) if norm_b else 0.0

    def _score_protocol_match(self, protocol_a: Dict[str, Any], details_b: Dict[str, Any]) -> float:
        """Score how well protocol matches ground truth details (0-1)."""
        if not protocol_a or not details_b:
            return 0.0
        
        # Check key protocol elements
        score = 0.0
        checks = 0
        
        # Check for runs/seeds
        if "num_runs" in protocol_a or "runs" in protocol_a:
            checks += 1
            if "runs" in str(details_b).lower() or "seeds" in str(details_b).lower():
                score += 1
        
        # Check for validation strategy
        if "validation_strategy" in protocol_a or "cv" in protocol_a:
            checks += 1
            if "validation" in str(details_b).lower() or "cross" in str(details_b).lower():
                score += 1
        
        # Check for hyperparameter tuning
        if "hyperparameter_tuning" in protocol_a:
            checks += 1
            if "hyperparameter" in str(details_b).lower() or "tuning" in str(details_b).lower():
                score += 1
        
        return score / checks if checks > 0 else 0.0