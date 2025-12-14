"""
Unified data models for experimental setup evaluation.
Replaces inconsistent dicts with typed dataclasses for better code clarity.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence levels for evaluation results."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class ComponentMatch:
    """Represents a match between generated and ground truth components."""
    generated: str
    ground_truth: str
    score: float
    method: str = "exact"  # exact, fuzzy, semantic
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentAlignment:
    """Alignment result for a specific component type (baselines, metrics, datasets)."""
    recall: float
    precision: float
    f1_score: float
    found: List[str]
    missing: List[str]
    extra: List[str]
    matches: List[ComponentMatch] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Standard result format for all evaluation operations."""
    overall_score: float
    component_scores: Dict[str, float]
    confidence: ConfidenceLevel
    details: Dict[str, Any] = field(default_factory=dict)
    feedback: Dict[str, List[str]] = field(default_factory=lambda: {
        "strengths": [],
        "weaknesses": [],
        "suggestions": []
    })


@dataclass
class QualityResult(EvaluationResult):
    """Quality scoring result with component breakdown."""
    completeness: float = 0.0
    diversity: float = 0.0
    alignment: float = 0.0
    datasets: float = 0.0
    protocol: float = 0.0


@dataclass
class AlignmentResult(EvaluationResult):
    """Alignment scoring result with component-specific alignments."""
    baseline_alignment: ComponentAlignment = field(default_factory=lambda: ComponentAlignment(0, 0, 0, [], [], []))
    metric_alignment: ComponentAlignment = field(default_factory=lambda: ComponentAlignment(0, 0, 0, [], [], []))
    dataset_alignment: ComponentAlignment = field(default_factory=lambda: ComponentAlignment(0, 0, 0, [], [], []))
    
    # New rigorous alignment
    hypotheses_alignment: float = 0.0
    control_alignment: float = 0.0
    protocol_alignment: float = 0.0
    
    overall_alignment: float = 0.0


@dataclass
class ComparisonResult:
    """Result of comparing two experimental setups."""
    winner: str  # "setup_a", "setup_b", "tie"
    quality_comparison: Dict[str, float]
    cost_comparison: Dict[str, float]
    component_overlap: Dict[str, Dict[str, float]]  # Jaccard similarities
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentalSetup:
    """Standardized experimental setup representation."""
    baselines: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    significance_tests: List[str] = field(default_factory=list)
    experimental_protocol: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    # New rigorous criteria
    hypotheses: List[str] = field(default_factory=list)
    sample_size_analysis: str = ""
    control_groups: List[str] = field(default_factory=list)
    randomization_blinding: str = ""
    validity_assessments: Dict[str, str] = field(default_factory=dict)

    def normalize_component_names(self, component_type: str) -> List[str]:
        """Normalize component names for comparison."""
        components = getattr(self, component_type, [])
        if not isinstance(components, list):
            return []
        return [self._normalize_name(name) for name in components]

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a component name."""
        if isinstance(name, dict):
            name = name.get("name", str(name))
        return str(name).lower().strip().replace("-", " ").replace("_", " ")


@dataclass
class PaperContext:
    """Paper context for experimental setup generation."""
    title: str
    abstract: str
    introduction: str
    related_work: str
    research_question: str
    domain: str
    method_description: str


@dataclass
class GroundTruth:
    """Ground truth experimental setup extracted from paper."""
    baselines: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    significance_tests: List[str] = field(default_factory=list)
    experimental_details: Dict[str, Any] = field(default_factory=dict)
    has_experiments: bool = False
    extraction_confidence: ConfidenceLevel = ConfidenceLevel.NONE


@dataclass
class GenerationMetadata:
    """Metadata for setup generation process."""
    paper_id: str
    timestamp: str
    model: str
    approach: str  # "non_agentic", "agentic"
    time_elapsed: float
    llm_calls: int
    quality_score: float = 0.0
    description: str = ""


@dataclass
class SetupGenerationResult:
    """Complete result from setup generation."""
    setup: ExperimentalSetup
    metadata: GenerationMetadata
    quality_result: Optional[QualityResult] = None
    alignment_result: Optional[AlignmentResult] = None


@dataclass
class PipelineResult:
    """Complete pipeline result for a single paper."""
    paper_id: str
    paper_context: PaperContext
    non_agentic_result: SetupGenerationResult
    agentic_result: SetupGenerationResult
    ground_truth: GroundTruth
    comparison_result: ComparisonResult
    success: bool = True
    error: Optional[str] = None


# Utility functions for data model operations

def extract_setup_from_dict(data: Dict[str, Any]) -> ExperimentalSetup:
    """Extract ExperimentalSetup from various dict formats."""
    # Handle nested structures
    if "setup" in data:
        data = data["setup"]
    if "ai_setup" in data:
        data = data["ai_setup"]

    # Extract baselines - handle both string arrays and object arrays
    baselines_raw = data.get("baselines", [])
    baselines = []
    for baseline in baselines_raw:
        if isinstance(baseline, dict) and "name" in baseline:
            baselines.append(baseline["name"])
        elif isinstance(baseline, str):
            baselines.append(baseline)

    # Extract metrics - handle both string arrays and object arrays
    metrics_raw = data.get("metrics", [])
    metrics = []
    for metric in metrics_raw:
        if isinstance(metric, dict) and "name" in metric:
            metrics.append(metric["name"])
        elif isinstance(metric, str):
            metrics.append(metric)

    # Extract datasets - handle both string arrays and object arrays
    datasets_raw = data.get("datasets", [])
    datasets = []
    for dataset in datasets_raw:
        if isinstance(dataset, dict) and "name" in dataset:
            datasets.append(dataset["name"])
        elif isinstance(dataset, str):
            datasets.append(dataset)
            
    # Extract control groups - handle both string arrays and object arrays
    controls_raw = data.get("control_groups", [])
    controls = []
    for control in controls_raw:
        if isinstance(control, dict) and "name" in control:
            controls.append(control["name"])
        elif isinstance(control, str):
            controls.append(control)

    return ExperimentalSetup(
        baselines=baselines,
        metrics=metrics,
        datasets=datasets,
        significance_tests=data.get("significance_tests", []),
        experimental_protocol=data.get("experimental_protocol", {}),
        reasoning=data.get("reasoning", ""),
        hypotheses=data.get("hypotheses", []),
        sample_size_analysis=data.get("sample_size_analysis", ""),
        control_groups=controls,
        randomization_blinding=data.get("randomization_blinding", ""),
        validity_assessments=data.get("validity_assessments", {})
    )


def calculate_component_alignment(
    generated: List[str],
    ground_truth: List[str],
    normalize: bool = True,
    semantic_threshold: float = 0.7
) -> ComponentAlignment:
    """
    Calculate alignment metrics between generated and ground truth components.
    
    Uses two-phase matching:
    1. Exact matches after normalization
    2. Fuzzy semantic matching for remaining items
    
    Args:
        generated: List of generated component names
        ground_truth: List of ground truth component names
        normalize: Whether to normalize names before comparison
        semantic_threshold: Minimum similarity score (0-1) for fuzzy matches
    
    Returns:
        ComponentAlignment with recall, precision, F1, and detailed match information
    """
    from difflib import SequenceMatcher
    
    if not ground_truth:  # No ground truth to compare against
        return ComponentAlignment(
            recall=0.0, precision=0.0, f1_score=0.0,
            found=[], missing=list(generated), extra=list(generated)
        )
    
    if not generated:
        return ComponentAlignment(
            recall=0.0, precision=0.0, f1_score=0.0,
            found=[], missing=list(ground_truth), extra=[]
        )
    
    # Normalize if requested
    if normalize:
        gen_normalized = [(ExperimentalSetup._normalize_name(name), name) for name in generated]
        gt_normalized = [(ExperimentalSetup._normalize_name(name), name) for name in ground_truth]
    else:
        gen_normalized = [(name, name) for name in generated]
        gt_normalized = [(name, name) for name in ground_truth]
    
    # Phase 1: Exact matches
    gen_dict = dict(gen_normalized)
    gt_dict = dict(gt_normalized)
    
    exact_matches = set(gen_dict.keys()) & set(gt_dict.keys())
    
    # Track matches with details
    matches: List[ComponentMatch] = []
    matched_gen = set()
    matched_gt = set()
    
    for norm_name in exact_matches:
        matches.append(ComponentMatch(
            generated=gen_dict[norm_name],
            ground_truth=gt_dict[norm_name],
            score=1.0,
            method="exact"
        ))
        matched_gen.add(norm_name)
        matched_gt.add(norm_name)
    
    # Phase 2: Fuzzy/semantic matching for remaining items
    remaining_gen = [norm_name for norm_name in gen_dict.keys() if norm_name not in matched_gen]
    remaining_gt = [norm_name for norm_name in gt_dict.keys() if norm_name not in matched_gt]
    
    for gt_norm in remaining_gt:
        best_match = None
        best_score = 0.0
        
        for gen_norm in remaining_gen:
            # Calculate string similarity using SequenceMatcher
            similarity = SequenceMatcher(None, gen_norm, gt_norm).ratio()
            
            if similarity > best_score and similarity >= semantic_threshold:
                best_score = similarity
                best_match = gen_norm
        
        if best_match:
            matches.append(ComponentMatch(
                generated=gen_dict[best_match],
                ground_truth=gt_dict[gt_norm],
                score=best_score,
                method="semantic",
                details={"similarity": best_score}
            ))
            matched_gen.add(best_match)
            matched_gt.add(gt_norm)
            remaining_gen.remove(best_match)
    
    # Calculate metrics
    found_count = len(matched_gt)
    recall = found_count / len(ground_truth) if ground_truth else 0.0
    precision = found_count / len(generated) if generated else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Build result lists (use original names, not normalized)
    found = [gt_dict[norm] for norm in matched_gt]
    missing = [gt_dict[norm] for norm in gt_dict.keys() if norm not in matched_gt]
    extra = [gen_dict[norm] for norm in gen_dict.keys() if norm not in matched_gen]
    
    return ComponentAlignment(
        recall=recall,
        precision=precision,
        f1_score=f1_score,
        found=found,
        missing=missing,
        extra=extra,
        matches=matches
    )