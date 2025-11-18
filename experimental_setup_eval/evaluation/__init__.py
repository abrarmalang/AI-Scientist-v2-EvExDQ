"""
Evaluation components for experimental setup assessment.
"""

from .setup_quality_scorer import SetupQualityScorer
from .comparator import SetupComparator, format_comparison_report
from .alignment_scorer import AlignmentScorer, format_alignment_report

__all__ = [
    "SetupQualityScorer",
    "SetupComparator",
    "format_comparison_report",
    "AlignmentScorer",
    "format_alignment_report"
]
