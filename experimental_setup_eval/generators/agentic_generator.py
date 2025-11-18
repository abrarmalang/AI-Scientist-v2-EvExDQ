"""
Agentic Setup Generator (Simplified POC)

Multi-stage iterative generation with quality-based selection.
Demonstrates agentic behavior without full Agent Manager complexity.

Architecture:
- Stage 1: Generate multiple complete setups
- Stage 2: Score each setup
- Stage 3: Select best setup
- Future: Can be replaced with full AI Scientist Agent Manager integration
"""

import time
from typing import Dict, Any, List
from datetime import datetime
import json

from ai_scientist.llm import create_client
from ai_scientist.treesearch.backend import query
from experimental_setup_eval.generators.setup_spec import experimental_setup_spec
from experimental_setup_eval.evaluation.setup_quality_scorer import SetupQualityScorer


class AgenticSetupGenerator:
    """
    Simplified agentic setup generator (POC).

    Approach:
    1. Generate N draft setups with varied prompts
    2. Score each draft using SetupQualityScorer
    3. Select best setup based on score
    4. Optional: Refine best setup

    This is a simplified version that demonstrates agentic behavior.
    Can be replaced with full Agent Manager integration later.
    """

    def __init__(
        self,
        model: str = "gpt-4o-2024-11-20",
        num_drafts: int = 3,
        refine_best: bool = True
    ):
        """
        Initialize agentic generator.

        Args:
            model: LLM model to use
            num_drafts: Number of draft setups to generate
            refine_best: Whether to refine the best draft
        """
        self.model = model
        self.client, self.model_name = create_client(model)
        self.num_drafts = num_drafts
        self.refine_best = refine_best
        self.scorer = SetupQualityScorer()

    def generate_setup(
        self,
        paper_context: Dict[str, Any],
        paper_id: str
    ) -> Dict[str, Any]:
        """
        Generate experimental setup using agentic approach.

        Args:
            paper_context: Extracted paper context
            paper_id: Unique paper identifier

        Returns:
            {
                "setup": {...},  # Best setup
                "metadata": {
                    "paper_id": str,
                    "timestamp": str,
                    "model": str,
                    "approach": "agentic",
                    "time_elapsed": float,
                    "llm_calls": int,
                    "num_drafts": int,
                    "best_score": float,
                    "description": str
                },
                "exploration_tree": [
                    {
                        "draft_id": int,
                        "setup": {...},
                        "score": float,
                        "component_scores": {...},
                        "is_best": bool
                    },
                    ...
                ]
            }
        """
        start_time = time.time()
        llm_calls = 0

        print(f"\n{'='*70}")
        print(f"AGENTIC SETUP GENERATION: {paper_id}")
        print(f"{'='*70}")

        # Stage 1: Generate multiple draft setups
        print(f"\n[Stage 1] Generating {self.num_drafts} draft setups...")
        drafts = []
        for i in range(self.num_drafts):
            print(f"  - Generating draft {i+1}/{self.num_drafts}...", end=" ")

            # Vary prompt focus for each draft
            prompt = self._create_varied_prompt(paper_context, draft_num=i)

            draft_setup = query(
                system_message="You are an expert in machine learning research methodology.",
                user_message=prompt,
                model=self.model_name,
                func_spec=experimental_setup_spec,
                temperature=0.8  # Higher temperature for diversity
            )
            llm_calls += 1

            # Score this draft
            score_result = self.scorer.score_setup(draft_setup, paper_context)

            drafts.append({
                "draft_id": i,
                "setup": draft_setup,
                "score": score_result["overall_score"],
                "component_scores": score_result["component_scores"],
                "feedback": score_result["feedback"],
                "is_best": False
            })

            print(f"Score: {score_result['overall_score']:.3f}")

        # Stage 2: Select best draft
        print(f"\n[Stage 2] Selecting best setup...")
        drafts_sorted = sorted(drafts, key=lambda x: x["score"], reverse=True)
        best_draft = drafts_sorted[0]
        best_draft["is_best"] = True

        print(f"  ✓ Best: Draft {best_draft['draft_id']} (score: {best_draft['score']:.3f})")

        # Stage 3: Refine best draft (optional)
        if self.refine_best and best_draft["score"] < 0.95:
            print(f"\n[Stage 3] Refining best setup (current score: {best_draft['score']:.3f})...")

            # Create refinement prompt using feedback
            refinement_prompt = self._create_refinement_prompt(
                paper_context,
                best_draft["setup"],
                best_draft["feedback"]
            )

            refined_setup = query(
                system_message="You are an expert in machine learning research methodology.",
                user_message=refinement_prompt,
                model=self.model_name,
                func_spec=experimental_setup_spec,
                temperature=0.7
            )
            llm_calls += 1

            # Score refined setup
            refined_score_result = self.scorer.score_setup(refined_setup, paper_context)

            print(f"  - Refined score: {refined_score_result['overall_score']:.3f}")

            # Use refined if better
            if refined_score_result["overall_score"] > best_draft["score"]:
                print(f"  ✓ Refinement improved score (+{refined_score_result['overall_score'] - best_draft['score']:.3f})")

                # Add refined version to tree
                drafts.append({
                    "draft_id": len(drafts),
                    "setup": refined_setup,
                    "score": refined_score_result["overall_score"],
                    "component_scores": refined_score_result["component_scores"],
                    "feedback": refined_score_result["feedback"],
                    "is_best": True,
                    "is_refinement": True
                })

                best_draft["is_best"] = False
                best_draft = drafts[-1]
            else:
                print(f"  × Refinement did not improve (keeping original)")

        elapsed_time = time.time() - start_time

        print(f"\n{'='*70}")
        print(f"FINAL BEST SCORE: {best_draft['score']:.3f}")
        print(f"Time: {elapsed_time:.1f}s | LLM Calls: {llm_calls}")
        print(f"{'='*70}\n")

        return {
            "setup": best_draft["setup"],
            "metadata": {
                "paper_id": paper_id,
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "approach": "agentic",
                "time_elapsed": elapsed_time,
                "llm_calls": llm_calls,
                "num_drafts": self.num_drafts,
                "best_score": best_draft["score"],
                "description": f"Generated {self.num_drafts} drafts, selected best (score: {best_draft['score']:.3f})"
            },
            "exploration_tree": drafts
        }

    def _create_varied_prompt(
        self,
        paper_context: Dict[str, Any],
        draft_num: int
    ) -> str:
        """
        Create varied prompts to encourage exploration.

        Each draft focuses on different aspects:
        - Draft 0: Balanced approach
        - Draft 1: Focus on baseline diversity
        - Draft 2: Focus on metric rigor
        - Draft 3+: Focus on dataset variety
        """
        title = paper_context.get("title", "Unknown Title")
        abstract = paper_context.get("abstract", "")
        research_question = paper_context.get("research_question", "")
        domain = paper_context.get("domain", "ML")
        method_description = paper_context.get("method_description", "")
        related_work = paper_context.get("related_work", "")
        introduction = paper_context.get("introduction", "")

        base_context = f"""You are an expert in machine learning research methodology. Design a comprehensive experimental setup for evaluating the research in this paper.

**Paper Title**: {title}

**Abstract**: {abstract}

**Research Question**: {research_question}

**Domain**: {domain}

**Introduction**: {introduction[:3000] if introduction else "Not available"}

**Related Work**: {related_work[:4000] if related_work else "Not available"}

**Method Description**: {method_description[:3000] if method_description else "Not available"}

---

"""

        # Vary focus based on draft number
        if draft_num == 0:
            focus = """**Focus for this setup**: Create a **balanced and comprehensive** experimental design that covers all aspects equally. Ensure completeness across baselines, metrics, datasets, and protocol.

**STEP 1 - SCAN RELATED WORK**: Before proposing any baselines, carefully read the Related Work section above. List out EVERY specific method/model name mentioned (e.g., "GraphCast", "Pangu-Weather", "BERT", "ResNet-50"). These are your primary baseline candidates.

**STEP 2 - USE EXACT NAMES**: When you find specific method names in Related Work, use those EXACT names as baselines. Do NOT generalize them (e.g., use "GraphCast" not "graph neural network model").

**STEP 3 - ADD DIVERSITY**: After including specific methods from Related Work, add 1-2 simple baselines for completeness."""

        elif draft_num == 1:
            focus = """**Focus for this setup**: Prioritize **baseline diversity**.

**CRITICAL - BASELINE EXTRACTION PROCESS**:
1. **READ Related Work section thoroughly** - It contains specific method names
2. **IDENTIFY at least 3-5 specific methods** mentioned by name (e.g., "Pangu-Weather", "GraphCast", "IFS")
3. **USE THOSE EXACT NAMES** as baselines - this is NON-NEGOTIABLE
4. **DO NOT use generic descriptions** like "numerical weather prediction models" - find the SPECIFIC model names
5. **Include varying complexity**: After specific methods from Related Work, add 1 simple baseline (e.g., "Persistence")

**Example of GOOD baselines**: "GraphCast", "Pangu-Weather", "ECMWF IFS", "Persistence Model"
**Example of BAD baselines**: "Graph neural network methods", "Transformer models", "Traditional NWP"""

        elif draft_num == 2:
            focus = """**Focus for this setup**: Prioritize **metric rigor and alignment**.

**CRITICAL - DOMAIN-SPECIFIC METRICS**:
1. **INFER from research question**: What specific aspect is being measured?
2. **USE domain-specific terminology**: Not "Accuracy" but "Forecast Accuracy", "Localization Accuracy", "Translation Accuracy"
3. **LOOK at the domain**: Weather forecasting uses RMSE/CRPS, NLP uses BLEU/perplexity, CV uses mAP/IoU
4. **INCLUDE both primary and diagnostic metrics**: What measures success + what provides insights

**Example**: For weather forecasting → "Forecast Accuracy", "RMSE", "CRPS" (NOT just "Accuracy", "Error Rate")"""

        else:
            focus = """**Focus for this setup**: Prioritize **dataset variety and appropriateness**.

**CRITICAL - DATASET IDENTIFICATION**:
1. **SCAN Introduction/Method** for mentions of data collection, dataset types, or benchmark names
2. **LOOK for specifics**: Dataset names ("WeatherBench 2"), data types ("RSSI fingerprints"), collection methods
3. **USE what's mentioned or implied** in paper context - don't hallucinate random datasets
4. **INCLUDE standard benchmarks** for this specific research domain if identified

**Example**: If paper mentions "WeatherBench 2" → include it. If method describes "indoor path data" → reference that."""

        guidelines = """
Design the experimental setup including:

1. **Baselines**: Specific comparison methods (use exact names from Related Work when available)
2. **Metrics**: Domain-specific evaluation metrics aligned with the research question
3. **Datasets**: Datasets mentioned or implied in the paper context
4. **Significance Tests**: Statistical tests for validation
5. **Experimental Protocol**: Reproducible procedure with multiple runs, seeds, validation strategy
6. **Reasoning**: Justify the overall setup design

**Critical Requirements**:
- Use SPECIFIC names from Related Work, not generic descriptions
- Use domain-specific metric names that match the research area
- Reference datasets or data types mentioned in the paper context
- Be concrete and specific, never vague
"""

        return base_context + focus + guidelines

    def _create_refinement_prompt(
        self,
        paper_context: Dict[str, Any],
        current_setup: Dict[str, Any],
        feedback: Dict[str, List[str]]
    ) -> str:
        """Create prompt to refine the best setup based on feedback."""

        title = paper_context.get("title", "Unknown Title")

        # Format current setup
        setup_json = json.dumps(current_setup, indent=2)

        # Format feedback
        weaknesses_str = "\n".join([f"  - {w}" for w in feedback.get("weaknesses", [])])
        suggestions_str = "\n".join([f"  - {s}" for s in feedback.get("suggestions", [])])

        prompt = f"""You are an expert in machine learning research methodology. You previously designed an experimental setup for the paper "{title}".

Your setup is good, but needs refinement based on the following feedback:

**Current Setup**:
```json
{setup_json}
```

**Weaknesses Identified**:
{weaknesses_str if weaknesses_str else "  (none)"}

**Suggestions for Improvement**:
{suggestions_str if suggestions_str else "  (none)"}

---

**Your Task**: Refine the experimental setup to address the weaknesses and incorporate the suggestions. Maintain the strengths of the current setup while improving the weak areas.

Focus on:
- Addressing each specific weakness
- Implementing the suggestions
- Maintaining or improving overall quality
- Keeping the setup concrete and specific

Generate the refined experimental setup.
"""
        return prompt
