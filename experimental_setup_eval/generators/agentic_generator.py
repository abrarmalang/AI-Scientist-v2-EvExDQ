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
# Enhanced backend with native Anthropic tool calling support
from experimental_setup_eval.backends.native_anthropic_backend import smart_query as query
from experimental_setup_eval.generators.setup_spec import experimental_setup_spec
from experimental_setup_eval.evaluation.unified_evaluator import UnifiedEvaluator
from experimental_setup_eval.core.config import get_default_config
from experimental_setup_eval.core.prompts import create_setup_prompt
from experimental_setup_eval.core.prompts import create_refinement_prompt


class AgenticSetupGenerator:
    """
    Simplified agentic setup generator (POC).

    Approach:
    1. Generate N draft setups with varied prompts
    2. Score each draft using UnifiedEvaluator
    3. Select best setup based on score
    4. Optional: Refine best setup

    This is a simplified version that demonstrates agentic behavior.
    Can be replaced with full Agent Manager integration later.
    """

    def __init__(
        self,
        model: str = "gpt-4o-2024-11-20",
        num_drafts: int = 3,
        refine_best: bool = True,
        evaluator: UnifiedEvaluator = None
    ):
        """
        Initialize agentic generator.

        Args:
            model: LLM model to use
            num_drafts: Number of draft setups to generate
            refine_best: Whether to refine the best draft
            evaluator: UnifiedEvaluator instance (optional)
        """
        self.model = model
        self.client, self.model_name = create_client(model)
        self.num_drafts = num_drafts
        self.refine_best = refine_best
        
        if evaluator:
            self.scorer = evaluator
        else:
            # Load default config if no evaluator provided
            config = get_default_config()
            self.scorer = UnifiedEvaluator(config.evaluation)

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
            score_result = self.scorer.score_quality(draft_setup, paper_context)

            drafts.append({
                "draft_id": i,
                "setup": draft_setup,
                "score": score_result.overall_score,
                "component_scores": score_result.component_scores,
                "feedback": score_result.feedback,
                "is_best": False
            })

            print(f"Score: {score_result.overall_score:.3f}")

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
            refined_score_result = self.scorer.score_quality(refined_setup, paper_context)

            print(f"  - Refined score: {refined_score_result.overall_score:.3f}")

            # Use refined if better
            if refined_score_result.overall_score > best_draft["score"]:
                print(f"  ✓ Refinement improved score (+{refined_score_result.overall_score - best_draft['score']:.3f})")

                # Add refined version to tree
                drafts.append({
                    "draft_id": len(drafts),
                    "setup": refined_setup,
                    "score": refined_score_result.overall_score,
                    "component_scores": refined_score_result.component_scores,
                    "feedback": refined_score_result.feedback,
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
        from experimental_setup_eval.core.prompts import create_setup_prompt
        
        # Map draft number to focus type
        if draft_num == 0:
            focus_type = "balanced"
        elif draft_num == 1:
            focus_type = "diversity"
        elif draft_num == 2:
            focus_type = "metrics"
        else:
            focus_type = "datasets"
            
        return create_setup_prompt(paper_context, focus_type)

    def _create_refinement_prompt(
        self,
        paper_context: Dict[str, Any],
        current_setup: Dict[str, Any],
        feedback: Dict[str, List[str]]
    ) -> str:
        """Create prompt to refine the best setup based on feedback."""
        from experimental_setup_eval.core.prompts import create_refinement_prompt
        return create_refinement_prompt(paper_context, current_setup, feedback)
