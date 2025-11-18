"""
Non-Agentic Setup Generator

Single-shot generation approach (baseline).
One LLM call to generate complete experimental setup.
"""

import time
from typing import Dict, Any
from datetime import datetime

from ai_scientist.llm import create_client
from ai_scientist.treesearch.backend import query
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

        # Extract key information from paper context
        title = paper_context.get("title", "Unknown Title")
        abstract = paper_context.get("abstract", "")
        research_question = paper_context.get("research_question", "")
        domain = paper_context.get("domain", "ML")
        method_description = paper_context.get("method_description", "")
        related_work = paper_context.get("related_work", "")
        introduction = paper_context.get("introduction", "")

        prompt = f"""You are an expert in machine learning research methodology. Your task is to design a comprehensive experimental setup for evaluating the research proposed in the following paper.

**Paper Title**: {title}

**Abstract**: {abstract}

**Research Question**: {research_question}

**Domain**: {domain}

**Introduction**: {introduction[:3000] if introduction else "Not available"}

**Related Work**: {related_work[:4000] if related_work else "Not available"}

**Method Description**: {method_description[:3000] if method_description else "Not available"}

---

**Your Task**: Design a complete experimental setup that includes:

1. **Baselines**: What baseline methods should be compared against?
   **CRITICAL**: Look carefully at the Related Work section above to identify specific prior methods mentioned by name.
   Include:
   - Specific prior methods mentioned in Related Work (use exact names if provided)
   - Simple baselines appropriate for this domain
   - Ablation baselines if applicable
   - Ensure diversity in complexity levels

   **Example**: If Related Work mentions "BERT" and "GPT-2", include those as baselines, not generic "transformer model".

2. **Metrics**: What evaluation metrics should be used?
   **CRITICAL**: Look at the research question and domain to infer what metrics the authors would likely use.
   Consider:
   - Primary metrics that directly measure the research question (be specific to the domain)
   - Diagnostic metrics for deeper analysis
   - Domain-appropriate metrics (e.g., for localization: accuracy, precision error; for NLP: perplexity, BLEU; for CV: mAP, IoU)
   - Specify if higher is better and expected ranges

   **Example**: For indoor localization, use "Localization Accuracy" not just "Accuracy".

3. **Datasets**: What datasets should be used for evaluation?
   **CRITICAL**: If the Introduction or Method mentions specific data collection or dataset types, reference those.
   Include:
   - Datasets mentioned or implied in the paper (e.g., "indoor paths", "RSSI fingerprints")
   - Standard benchmark datasets used in this specific research area
   - Consider dataset characteristics mentioned in the method (size, domain, split)

4. **Significance Tests**: What statistical tests should validate the results?
   - Paired t-test, ANOVA, Wilcoxon, etc.
   - Specify when each test is appropriate

5. **Experimental Protocol**: Define the experimental procedure:
   - Number of runs with different random seeds
   - Cross-validation strategy
   - Hyperparameter tuning approach
   - Estimated compute budget

6. **Overall Reasoning**: Explain why this experimental setup is appropriate for evaluating the proposed research.

**Important Guidelines**:
- Design the setup to be rigorous, reproducible, and comprehensive
- Ensure baselines cover different complexity levels
- Choose metrics that align with the paper's research question
- Select datasets appropriate for the domain
- Specify concrete details (not vague descriptions)
- Justify your choices with clear rationale

Generate a complete experimental setup following these guidelines.
"""
        return prompt
