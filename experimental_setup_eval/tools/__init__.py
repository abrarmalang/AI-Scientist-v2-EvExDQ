"""
Tools for experimental setup evaluation.

This module provides tools that extend AI Scientist v2's BaseTool framework.
"""

from .paper_processor import PaperProcessorTool
from .ground_truth_extractor import GroundTruthExtractor

__all__ = [
    "PaperProcessorTool",
    "GroundTruthExtractor"
]
