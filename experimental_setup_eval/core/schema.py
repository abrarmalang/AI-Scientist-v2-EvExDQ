"""
OmegaConf-based configuration schema for experimental setup evaluation.
Follows AI Scientist's proven configuration patterns with typed dataclasses.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration with available models and selection logic."""
    default: str = "llama-3.1-8b-instant"
    available: Dict[str, str] = field(default_factory=lambda: {
        # Groq Models (Fast and cost-effective)
        "llama-3.1-8b": "llama-3.1-8b-instant",
        "llama-3.1-70b": "llama-3.1-70b-versatile",
        # OpenAI Models
        "gpt-4o": "gpt-4o-2024-11-20",
        "gpt-4-turbo": "gpt-4-turbo",
        # Anthropic Models
        "claude-sonnet": "claude-sonnet-4-5-20250929",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-haiku": "claude-3-haiku-20240307",
    })


@dataclass
class QualityWeights:
    """Weights for quality scoring components."""
    completeness: float = 0.30
    diversity: float = 0.20
    alignment: float = 0.25
    datasets: float = 0.15
    protocol: float = 0.10


@dataclass
class AlignmentWeights:
    """Weights for alignment scoring."""
    semantic_weight: float = 0.6
    exact_weight: float = 0.4
    baseline_weight: float = 0.4
    metric_weight: float = 0.3
    dataset_weight: float = 0.3


@dataclass
class AgenticConfig:
    """Configuration for agentic generator."""
    num_drafts: int = 3
    max_iterations: int = 5
    improvement_threshold: float = 0.05


@dataclass
class EvaluationConfig:
    """Evaluation settings and scoring configuration."""
    quality_weights: QualityWeights = field(default_factory=QualityWeights)
    alignment_weights: AlignmentWeights = field(default_factory=AlignmentWeights)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)

    # Quality thresholds for classification
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "excellent": 0.8,
        "good": 0.6,
        "fair": 0.4,
    })

    # Similarity thresholds for semantic matching
    similarity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4,
    })


@dataclass
class PipelineConfig:
    """Pipeline execution configuration."""
    redact_sections: List[str] = field(default_factory=lambda: [
        "experiment", "result", "evaluation"
    ])
    output_pattern: str = "runs/{timestamp}_full_eval"
    max_retries: int = 3
    timeout_seconds: int = 300
    parallel_processing: bool = False


@dataclass
class CollectionConfig:
    """Configuration for paper collection."""
    max_results: int = 10
    min_citations: int = 0
    year_range: str = "2020-2025"
    download_pdfs: bool = True
    convert_markdown: bool = True
    fields_of_study: List[str] = field(default_factory=lambda: ["Computer Science"])


@dataclass
class ValidationConfig:
    """Configuration for paper validation."""
    min_title_length: int = 15
    min_abstract_length: int = 100
    min_intro_length: int = 300
    min_method_length: int = 200
    min_total_length: int = 1000


@dataclass
class EvalSetupConfig:
    """Main configuration container for experimental setup evaluation."""
    models: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Global settings
    debug: bool = False
    verbose: bool = True
    save_intermediate: bool = True