"""
Alignment Scorer

Compares generated experimental setup against ground truth.
Calculates how well AI predictions match human choices.
"""

from typing import Dict, List, Set, Any


class AlignmentScorer:
    """
    Score alignment between generated setup and ground truth.

    Metrics:
    - Baseline recall: % of true baselines found
    - Metric recall: % of true metrics found
    - Dataset recall: % of true datasets found
    - Overall alignment: weighted average
    """

    def score_alignment(
        self,
        generated_setup: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score how well generated setup aligns with ground truth.

        Args:
            generated_setup: AI-generated setup (from generator)
            ground_truth: Extracted ground truth (from extractor)

        Returns:
            {
                "baseline_alignment": {
                    "recall": float,
                    "found": List[str],
                    "missing": List[str],
                    "extra": List[str]
                },
                "metric_alignment": {...},
                "dataset_alignment": {...},
                "overall_alignment": float,
                "confidence": "high" | "medium" | "low"
            }
        """
        # Extract generated components
        if "ai_setup" in generated_setup:
            generated_setup = generated_setup["ai_setup"]
        elif "setup" in generated_setup:
            generated_setup = generated_setup["setup"]
            if "ai_setup" in generated_setup:
                generated_setup = generated_setup["ai_setup"]

        gen_baselines = self._extract_names(generated_setup.get("baselines", []))
        gen_metrics = self._extract_names(generated_setup.get("metrics", []))
        gen_datasets = self._extract_names(generated_setup.get("datasets", []))

        # Extract ground truth components
        gt_baselines = self._extract_names(ground_truth.get("baselines", []))
        gt_metrics = self._extract_names(ground_truth.get("metrics", []))
        gt_datasets = self._extract_names(ground_truth.get("datasets", []))

        # Check if ground truth exists
        has_gt = (len(gt_baselines) + len(gt_metrics) + len(gt_datasets)) > 0
        confidence = ground_truth.get("extraction_confidence", "low")

        if not has_gt:
            return {
                "baseline_alignment": self._empty_alignment(),
                "metric_alignment": self._empty_alignment(),
                "dataset_alignment": self._empty_alignment(),
                "overall_alignment": 0.0,
                "confidence": "none",
                "note": "No ground truth available for comparison"
            }

        # Score each component
        baseline_align = self._score_component(gen_baselines, gt_baselines)
        metric_align = self._score_component(gen_metrics, gt_metrics)
        dataset_align = self._score_component(gen_datasets, gt_datasets)

        # Calculate overall alignment (weighted average)
        # Weight: baselines=40%, metrics=40%, datasets=20%
        weights = {"baselines": 0.4, "metrics": 0.4, "datasets": 0.2}

        overall = 0.0
        total_weight = 0.0

        if gt_baselines:
            overall += weights["baselines"] * baseline_align["recall"]
            total_weight += weights["baselines"]

        if gt_metrics:
            overall += weights["metrics"] * metric_align["recall"]
            total_weight += weights["metrics"]

        if gt_datasets:
            overall += weights["datasets"] * dataset_align["recall"]
            total_weight += weights["datasets"]

        if total_weight > 0:
            overall = overall / total_weight
        else:
            overall = 0.0

        return {
            "baseline_alignment": baseline_align,
            "metric_alignment": metric_align,
            "dataset_alignment": dataset_align,
            "overall_alignment": overall,
            "confidence": confidence,
            "ground_truth_counts": {
                "baselines": len(gt_baselines),
                "metrics": len(gt_metrics),
                "datasets": len(gt_datasets)
            }
        }

    def _extract_names(self, components: List[Dict]) -> Set[str]:
        """Extract normalized names from component list."""
        names = set()
        for comp in components:
            if isinstance(comp, dict):
                name = comp.get("name", "")
            else:
                name = str(comp)

            # Normalize
            name = name.lower().strip()

            # Remove common suffixes/prefixes
            name = name.replace("baseline", "").replace("method", "").strip()

            if name:
                names.add(name)

        return names

    def _score_component(
        self,
        generated: Set[str],
        ground_truth: Set[str]
    ) -> Dict[str, Any]:
        """Score alignment for one component type."""

        if not ground_truth:
            return self._empty_alignment()

        # Find matches (fuzzy matching)
        found = set()
        missing = set(ground_truth)

        for gen_name in generated:
            for gt_name in ground_truth:
                if self._fuzzy_match(gen_name, gt_name):
                    found.add(gt_name)
                    if gt_name in missing:
                        missing.remove(gt_name)

        # Recall = found / total
        recall = len(found) / len(ground_truth) if ground_truth else 0.0

        # Extra = generated but not in ground truth
        extra = generated - found

        return {
            "recall": recall,
            "found": sorted(list(found)),
            "missing": sorted(list(missing)),
            "extra": sorted(list(extra)),
            "found_count": len(found),
            "total_count": len(ground_truth)
        }

    def _fuzzy_match(self, name1: str, name2: str) -> bool:
        """Check if two names match (fuzzy)."""
        name1 = name1.lower().strip()
        name2 = name2.lower().strip()

        # Exact match
        if name1 == name2:
            return True

        # Substring match
        if name1 in name2 or name2 in name1:
            return True

        # Common abbreviations
        abbrev_map = {
            "mse": "mean squared error",
            "mae": "mean absolute error",
            "rmse": "root mean squared error",
            "f1": "f1 score",
            "auc": "area under curve",
            "sota": "state-of-the-art",
            "bert": "bidirectional encoder representations",
            "gpt": "generative pre-trained transformer"
        }

        for abbrev, full in abbrev_map.items():
            if (abbrev in name1 and full in name2) or (abbrev in name2 and full in name1):
                return True

        return False

    def _empty_alignment(self) -> Dict[str, Any]:
        """Return empty alignment result."""
        return {
            "recall": 0.0,
            "found": [],
            "missing": [],
            "extra": [],
            "found_count": 0,
            "total_count": 0
        }


def format_alignment_report(alignment: Dict[str, Any], label: str = "Generated") -> str:
    """Format alignment scores as readable text."""

    if alignment.get("confidence") == "none":
        return f"\n{label} ALIGNMENT: Cannot be computed (no ground truth)\n"

    report = f"""
{'='*70}
{label.upper()} ALIGNMENT WITH GROUND TRUTH
{'='*70}

Overall Alignment: {alignment['overall_alignment']:.1%}
Confidence: {alignment.get('confidence', 'unknown').upper()}

{'─'*70}
BASELINES
{'─'*70}
  Recall: {alignment['baseline_alignment']['recall']:.1%} ({alignment['baseline_alignment']['found_count']}/{alignment['baseline_alignment']['total_count']})

  Found: {', '.join(alignment['baseline_alignment']['found']) if alignment['baseline_alignment']['found'] else '(none)'}

  Missing: {', '.join(alignment['baseline_alignment']['missing']) if alignment['baseline_alignment']['missing'] else '(none)'}

  Extra (not in GT): {', '.join(alignment['baseline_alignment']['extra'][:3]) if alignment['baseline_alignment']['extra'] else '(none)'}

{'─'*70}
METRICS
{'─'*70}
  Recall: {alignment['metric_alignment']['recall']:.1%} ({alignment['metric_alignment']['found_count']}/{alignment['metric_alignment']['total_count']})

  Found: {', '.join(alignment['metric_alignment']['found']) if alignment['metric_alignment']['found'] else '(none)'}

  Missing: {', '.join(alignment['metric_alignment']['missing']) if alignment['metric_alignment']['missing'] else '(none)'}

{'─'*70}
DATASETS
{'─'*70}
  Recall: {alignment['dataset_alignment']['recall']:.1%} ({alignment['dataset_alignment']['found_count']}/{alignment['dataset_alignment']['total_count']})

  Found: {', '.join(alignment['dataset_alignment']['found']) if alignment['dataset_alignment']['found'] else '(none)'}

  Missing: {', '.join(alignment['dataset_alignment']['missing']) if alignment['dataset_alignment']['missing'] else '(none)'}

{'='*70}
"""
    return report
