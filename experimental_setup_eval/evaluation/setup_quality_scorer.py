"""
Setup Quality Scorer

Evaluates experimental setup quality WITHOUT seeing ground truth.
Used by agentic generator to score nodes during tree search.
"""

from typing import Dict, Any, List
import json
from pathlib import Path


class SetupQualityScorer:
    """
    Score setup quality on 0-1 scale without ground truth.

    Components:
    - Completeness (30%): Has all required components?
    - Baseline Diversity (20%): Multiple baseline types?
    - Metric Alignment (25%): Metrics match paper domain?
    - Dataset Appropriateness (15%): Datasets match domain?
    - Protocol Rigor (10%): Proper experimental protocol?
    """

    def __init__(
        self,
        weights: Dict[str, float] = None
    ):
        """
        Initialize scorer with component weights.

        Args:
            weights: Dictionary of component weights (must sum to 1.0)
                    Default: {
                        "completeness": 0.30,
                        "diversity": 0.20,
                        "alignment": 0.25,
                        "datasets": 0.15,
                        "protocol": 0.10
                    }
        """
        self.weights = weights or {
            "completeness": 0.30,
            "diversity": 0.20,
            "alignment": 0.25,
            "datasets": 0.15,
            "protocol": 0.10
        }

        # Validate weights sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def score_setup(
        self,
        setup: Dict[str, Any],
        paper_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score a setup's quality.

        Args:
            setup: Setup dictionary with baselines, metrics, datasets, protocol
            paper_context: Paper context for domain/task alignment

        Returns:
            {
                "overall_score": float (0-1),
                "component_scores": {
                    "completeness": float,
                    "diversity": float,
                    "alignment": float,
                    "datasets": float,
                    "protocol": float
                },
                "feedback": {
                    "strengths": List[str],
                    "weaknesses": List[str],
                    "suggestions": List[str]
                }
            }
        """
        # Extract AI setup if nested
        if "ai_setup" in setup:
            setup = setup["ai_setup"]

        # Score each component
        completeness = self._score_completeness(setup)
        diversity = self._score_baseline_diversity(setup)
        alignment = self._score_metric_alignment(setup, paper_context)
        datasets = self._score_dataset_appropriateness(setup, paper_context)
        protocol = self._score_protocol_rigor(setup)

        component_scores = {
            "completeness": completeness,
            "diversity": diversity,
            "alignment": alignment,
            "datasets": datasets,
            "protocol": protocol
        }

        # Weighted overall score
        overall_score = sum(
            self.weights[component] * score
            for component, score in component_scores.items()
        )

        # Generate feedback
        feedback = self._generate_feedback(component_scores, setup, paper_context)

        return {
            "overall_score": overall_score,
            "component_scores": component_scores,
            "feedback": feedback
        }

    def _score_completeness(self, setup: Dict) -> float:
        """
        Score setup completeness (0-1).

        Required components:
        - baselines (at least 1)
        - metrics (at least 1)
        - datasets (at least 1)
        - experimental_protocol with num_runs
        - reasoning
        """
        score = 0.0
        max_score = 5.0

        # Has baselines?
        if "baselines" in setup and len(setup["baselines"]) > 0:
            score += 1.0
            # Bonus for multiple baselines
            if len(setup["baselines"]) >= 3:
                score += 0.5

        # Has metrics?
        if "metrics" in setup and len(setup["metrics"]) > 0:
            score += 1.0
            # Bonus for multiple metrics
            if len(setup["metrics"]) >= 3:
                score += 0.5

        # Has datasets?
        if "datasets" in setup and len(setup["datasets"]) > 0:
            score += 1.0

        # Has protocol?
        if "experimental_protocol" in setup:
            protocol = setup["experimental_protocol"]
            if "num_runs" in protocol and protocol["num_runs"] >= 1:
                score += 1.0

        # Has reasoning?
        if "reasoning" in setup and len(setup.get("reasoning", "")) > 50:
            score += 0.5

        # Has significance tests?
        if "significance_tests" in setup and len(setup["significance_tests"]) > 0:
            score += 0.5

        return min(score / max_score, 1.0)

    def _score_baseline_diversity(self, setup: Dict) -> float:
        """
        Score baseline diversity (0-1).

        Good diversity includes:
        - Simple baseline (random, majority, etc.)
        - SOTA or previous work baseline
        - Ablation or intermediate complexity baseline
        """
        if "baselines" not in setup or len(setup["baselines"]) == 0:
            return 0.0

        baselines = setup["baselines"]
        score = 0.0

        # Count baselines
        num_baselines = len(baselines)
        if num_baselines == 0:
            return 0.0
        elif num_baselines == 1:
            score += 0.3
        elif num_baselines == 2:
            score += 0.5
        elif num_baselines >= 3:
            score += 0.7

        # Check for complexity diversity
        complexity_keywords = {
            "simple": ["random", "majority", "simple", "naive", "trivial"],
            "moderate": ["previous", "baseline", "standard", "shallow"],
            "complex": ["sota", "state-of-the-art", "advanced", "deep", "large"]
        }

        found_complexities = set()
        for baseline in baselines:
            name_lower = baseline.get("name", "").lower()
            desc_lower = baseline.get("description", "").lower()
            text = name_lower + " " + desc_lower

            for complexity, keywords in complexity_keywords.items():
                if any(kw in text for kw in keywords):
                    found_complexities.add(complexity)

        # Bonus for having different complexity levels
        diversity_bonus = len(found_complexities) * 0.1
        score += diversity_bonus

        return min(score, 1.0)

    def _score_metric_alignment(self, setup: Dict, paper_context: Dict) -> float:
        """
        Score metric alignment with paper domain (0-1).

        Checks:
        - Metrics have descriptions and rationale
        - Metrics specify higher_is_better
        - Domain-appropriate metrics (e.g., accuracy for classification)
        """
        if "metrics" not in setup or len(setup["metrics"]) == 0:
            return 0.0

        metrics = setup["metrics"]
        score = 0.0

        # Base score for having metrics
        num_metrics = len(metrics)
        if num_metrics >= 1:
            score += 0.3
        if num_metrics >= 3:
            score += 0.2

        # Check metric completeness
        complete_metrics = 0
        for metric in metrics:
            has_name = "name" in metric and len(metric["name"]) > 0
            has_desc = "description" in metric and len(metric["description"]) > 10
            has_rationale = "rationale" in metric and len(metric["rationale"]) > 20
            has_direction = "higher_is_better" in metric

            if has_name and has_desc and has_rationale and has_direction:
                complete_metrics += 1

        if num_metrics > 0:
            completeness_score = complete_metrics / num_metrics
            score += 0.3 * completeness_score

        # Domain alignment (basic heuristic)
        domain = paper_context.get("domain", "OTHER-ML").upper()
        metric_names = [m.get("name", "").lower() for m in metrics]

        # Classification metrics
        if "CLASSIFICATION" in domain or "IMAGE" in domain:
            if any(kw in " ".join(metric_names) for kw in ["accuracy", "f1", "precision", "recall", "auc"]):
                score += 0.2

        # Regression metrics
        elif "REGRESSION" in domain or "FORECASTING" in domain:
            if any(kw in " ".join(metric_names) for kw in ["mse", "mae", "rmse", "r2", "r-squared"]):
                score += 0.2

        # Generation metrics
        elif "GENERATION" in domain or "NLP" in domain:
            if any(kw in " ".join(metric_names) for kw in ["bleu", "rouge", "perplexity", "accuracy"]):
                score += 0.2

        # Default: any metric is better than none
        else:
            if len(metric_names) > 0:
                score += 0.1

        return min(score, 1.0)

    def _score_dataset_appropriateness(self, setup: Dict, paper_context: Dict) -> float:
        """
        Score dataset appropriateness (0-1).

        Checks:
        - Has at least one dataset
        - Datasets have rationale
        - Datasets have size and split info
        """
        if "datasets" not in setup or len(setup["datasets"]) == 0:
            return 0.0

        datasets = setup["datasets"]
        score = 0.0

        # Base score for having datasets
        num_datasets = len(datasets)
        if num_datasets >= 1:
            score += 0.4
        if num_datasets >= 2:
            score += 0.2

        # Check dataset completeness
        complete_datasets = 0
        for dataset in datasets:
            has_name = "name" in dataset and len(dataset["name"]) > 0
            has_desc = "description" in dataset and len(dataset["description"]) > 10
            has_rationale = "rationale" in dataset and len(dataset["rationale"]) > 20
            has_size = "size" in dataset
            has_splits = "splits" in dataset

            completeness = sum([has_name, has_desc, has_rationale, has_size, has_splits])
            complete_datasets += completeness / 5.0

        if num_datasets > 0:
            completeness_score = complete_datasets / num_datasets
            score += 0.4 * completeness_score

        return min(score, 1.0)

    def _score_protocol_rigor(self, setup: Dict) -> float:
        """
        Score experimental protocol rigor (0-1).

        Checks:
        - Multiple runs with different seeds
        - Cross-validation strategy
        - Hyperparameter tuning approach
        - Compute budget specified
        """
        if "experimental_protocol" not in setup:
            return 0.0

        protocol = setup["experimental_protocol"]
        score = 0.0

        # Number of runs
        num_runs = protocol.get("num_runs", 0)
        if num_runs >= 1:
            score += 0.2
        if num_runs >= 3:
            score += 0.1
        if num_runs >= 5:
            score += 0.1

        # Random seeds specified
        if "random_seeds" in protocol and len(protocol["random_seeds"]) >= num_runs:
            score += 0.2

        # Cross-validation
        if "cross_validation" in protocol and len(protocol["cross_validation"]) > 10:
            score += 0.2

        # Hyperparameter tuning
        if "hyperparameter_tuning" in protocol and len(protocol["hyperparameter_tuning"]) > 10:
            score += 0.2

        # Compute budget
        if "compute_budget" in protocol:
            score += 0.1

        return min(score, 1.0)

    def _generate_feedback(
        self,
        component_scores: Dict[str, float],
        setup: Dict,
        paper_context: Dict
    ) -> Dict[str, List[str]]:
        """Generate actionable feedback based on scores."""
        strengths = []
        weaknesses = []
        suggestions = []

        # Completeness feedback
        if component_scores["completeness"] >= 0.8:
            strengths.append("Setup has all essential components (baselines, metrics, datasets, protocol)")
        elif component_scores["completeness"] < 0.5:
            weaknesses.append("Setup is missing key components")
            suggestions.append("Ensure setup includes baselines, metrics, datasets, and experimental protocol")

        # Diversity feedback
        if component_scores["diversity"] >= 0.7:
            strengths.append("Good baseline diversity across complexity levels")
        elif component_scores["diversity"] < 0.5:
            weaknesses.append("Limited baseline diversity")
            suggestions.append("Add baselines of varying complexity (simple, moderate, SOTA)")

        # Alignment feedback
        if component_scores["alignment"] >= 0.7:
            strengths.append("Metrics align well with paper's domain and task")
        elif component_scores["alignment"] < 0.5:
            weaknesses.append("Metrics may not align with paper's domain")
            suggestions.append("Choose metrics appropriate for the paper's task (classification/regression/generation)")

        # Dataset feedback
        if component_scores["datasets"] >= 0.7:
            strengths.append("Datasets are well-specified and appropriate")
        elif component_scores["datasets"] < 0.5:
            weaknesses.append("Dataset specifications incomplete")
            suggestions.append("Provide dataset rationale, size, and train/val/test splits")

        # Protocol feedback
        if component_scores["protocol"] >= 0.7:
            strengths.append("Rigorous experimental protocol with multiple runs and proper validation")
        elif component_scores["protocol"] < 0.5:
            weaknesses.append("Experimental protocol lacks rigor")
            suggestions.append("Specify multiple runs with different seeds, cross-validation, and tuning strategy")

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "suggestions": suggestions
        }


def score_setup_file(setup_path: str, context_path: str, weights: Dict = None) -> Dict:
    """
    Convenience function to score setup from files.

    Args:
        setup_path: Path to setup JSON file
        context_path: Path to paper context JSON file
        weights: Optional custom weights

    Returns:
        Scoring results dictionary
    """
    with open(setup_path, 'r') as f:
        setup = json.load(f)

    with open(context_path, 'r') as f:
        paper_context = json.load(f)

    scorer = SetupQualityScorer(weights=weights)
    return scorer.score_setup(setup, paper_context)
