"""
Non-Agentic Setup Generator

Single-shot generation approach (baseline).
One LLM call to generate complete experimental setup.
"""

import time
from typing import Dict, Any
from datetime import datetime

from ai_scientist.llm import create_client
# Enhanced backend with native Anthropic tool calling support
from experimental_setup_eval.backends.native_anthropic_backend import smart_query as query
from experimental_setup_eval.generators.setup_spec import experimental_setup_spec


class NonAgenticSetupGenerator:
    """
    Baseline setup generator using single LLM call.

    Fast, cheap, simple approach without iteration or refinement.
    """

    def __init__(self, model: str = "gpt-4o-2024-11-20"):
        """
        Initialize generator.

        Args:
            model: LLM model to use for generation
        """
        self.model = model
        self.client, self.model_name = create_client(model)

    def generate_setup(
        self,
        paper_context: Dict[str, Any],
        paper_id: str
    ) -> Dict[str, Any]:
        """
        Generate experimental setup with single LLM call.

        Args:
            paper_context: Extracted paper context (research question, domain, etc.)
            paper_id: Unique paper identifier

        Returns:
            {
                "setup": {
                    "baselines": [...],
                    "metrics": [...],
                    "datasets": [...],
                    "significance_tests": [...],
                    "experimental_protocol": {...},
                    "reasoning": str
                },
                "metadata": {
                    "paper_id": str,
                    "timestamp": str,
                    "model": str,
                    "approach": "non_agentic",
                    "time_elapsed": float,
                    "llm_calls": int
                }
            }
        """
        start_time = time.time()

        # Create prompt for setup generation
        prompt = self._create_prompt(paper_context)

        # Single LLM call with function calling
        response = query(
            system_message="You are an expert in machine learning research methodology.",
            user_message=prompt,
            model=self.model_name,
            func_spec=experimental_setup_spec,
            temperature=0.7
        )

        elapsed_time = time.time() - start_time

        return {
            "setup": response,
            "metadata": {
                "paper_id": paper_id,
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "approach": "non_agentic",
                "time_elapsed": elapsed_time,
                "llm_calls": 1,
                "description": "Single-shot generation without iteration"
            }
        }

    def _create_prompt(self, paper_context: Dict[str, Any]) -> str:
        """Create prompt for setup generation."""
        from experimental_setup_eval.core.prompts import create_setup_prompt
        return create_setup_prompt(paper_context, focus_type="generic")
