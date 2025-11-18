"""
Experimental Setup Evaluation

A peer-level extension to AI Scientist v2 for evaluating experimental design quality.

This module evaluates how well AI can design experimental setups by:
1. Extracting research context (without experimental details)
2. Generating setups using non-agentic and agentic approaches
3. Comparing AI predictions against human choices (ground truth)

Architecture:
- tools/: Core processing tools (paper processor, ground truth extractor, suitability checker)
- generators/: Setup generation approaches (non-agentic, agentic)
- evaluation/: Scoring and comparison (quality scorer, comparator, alignment scorer)
- data_collection/: Paper collection utilities

Uses AI Scientist v2 as a dependency:
- ai_scientist.tools.base_tool.BaseTool
- ai_scientist.treesearch.backend (query, FunctionSpec)
- ai_scientist.llm (create_client, get_response_from_llm)
"""

__version__ = "0.1.0"
__author__ = "AI Scientist Extension Project"

# Main components
from .tools import PaperProcessorTool, GroundTruthExtractor
from .generators import NonAgenticSetupGenerator, AgenticSetupGenerator
from .evaluation import (
    SetupQualityScorer,
    SetupComparator,
    AlignmentScorer
)

__all__ = [
    # Tools
    "PaperProcessorTool",
    "GroundTruthExtractor",

    # Generators
    "NonAgenticSetupGenerator",
    "AgenticSetupGenerator",

    # Evaluation
    "SetupQualityScorer",
    "SetupComparator",
    "AlignmentScorer"
]
