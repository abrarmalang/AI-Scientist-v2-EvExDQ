"""
Unified evaluation components for experimental setup assessment.
Provides clean, simplified interfaces replacing multiple separate evaluator classes.
"""

from .unified_evaluator import UnifiedEvaluator
from .data_models import (
    EvaluationResult, QualityResult, AlignmentResult, ComparisonResult,
    ExperimentalSetup, PaperContext, GroundTruth, ComponentAlignment,
    ConfidenceLevel, extract_setup_from_dict, calculate_component_alignment
)

__all__ = [
    "UnifiedEvaluator",
    "EvaluationResult",
    "QualityResult",
    "AlignmentResult",
    "ComparisonResult",
    "ExperimentalSetup",
    "PaperContext",
    "GroundTruth",
    "ComponentAlignment",
    "ConfidenceLevel",
    "extract_setup_from_dict",
    "calculate_component_alignment"
]
