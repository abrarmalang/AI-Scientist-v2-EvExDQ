"""
Ground Truth Extractor

Extracts actual experimental setup from paper's experimental sections.
Uses LLM to parse baselines, metrics, and datasets from text.
"""

import json
from typing import Dict, List, Any

from ai_scientist.llm import create_client, get_response_from_llm


class GroundTruthExtractor:
    """
    Extract experimental setup components from paper text.

    Extracts:
    - Baselines: Methods compared against
    - Metrics: Evaluation metrics used
    - Datasets: Datasets evaluated on
    """

    def __init__(self, model: str = "gpt-4o-2024-11-20"):
        """Initialize extractor with LLM model."""
        self.model = model
        self.client, self.model_name = create_client(model)

    def extract_from_paper(self, paper_markdown: str) -> Dict[str, Any]:
        """
        Extract ground truth experimental setup from full paper.

        Args:
            paper_markdown: Full paper text in markdown

        Returns:
            {
                "baselines": [{"name": str, "description": str}, ...],
                "metrics": [{"name": str, "description": str}, ...],
                "datasets": [{"name": str, "description": str}, ...],
                "has_experiments": bool,
                "extraction_confidence": "high" | "medium" | "low"
            }
        """
        # Create extraction prompt
        prompt = self._create_extraction_prompt(paper_markdown)

        # Call LLM
        system_message = "You are an expert at extracting experimental setup details from machine learning papers."

        try:
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.model_name,
                system_message=system_message,
                temperature=0.3  # Low temperature for consistent extraction
            )

            # Parse response
            ground_truth = self._parse_response(response)
            return ground_truth

        except Exception as e:
            print(f"Warning: Could not extract ground truth: {e}")
            return {
                "baselines": [],
                "metrics": [],
                "datasets": [],
                "has_experiments": False,
                "extraction_confidence": "low",
                "error": str(e)
            }

    def _create_extraction_prompt(self, paper_text: str) -> str:
        """Create prompt for ground truth extraction."""

        # Truncate if too long (keep experimental sections)
        if len(paper_text) > 15000:
            # Try to find experimental sections
            sections = ["experiment", "evaluation", "result", "method", "setup"]
            relevant_parts = []

            lines = paper_text.split('\n')
            current_section = []
            in_relevant = False

            for line in lines:
                line_lower = line.lower()

                # Check if this is a section header
                if any(f'# {sec}' in line_lower or f'## {sec}' in line_lower for sec in sections):
                    if current_section:
                        relevant_parts.extend(current_section)
                        current_section = []
                    in_relevant = True

                if in_relevant:
                    current_section.append(line)
                    if len('\n'.join(relevant_parts + current_section)) > 12000:
                        break

            if relevant_parts:
                paper_text = '\n'.join(relevant_parts + current_section)
            else:
                # Just take first 12000 chars
                paper_text = paper_text[:12000]

        prompt = f"""Extract the experimental setup from this machine learning paper.

**Paper Text**:
{paper_text}

---

**Task**: Identify and extract:

1. **Baselines**: What methods did the authors compare their approach against?
   - Look for phrases like: "compared to", "baseline", "prior work", "we compare against"
   - Extract method names and brief descriptions

2. **Metrics**: What evaluation metrics did they use?
   - Look for: "evaluated using", "metrics", "measured by", "performance"
   - Extract metric names (e.g., accuracy, F1, MSE, BLEU)

3. **Datasets**: What datasets did they evaluate on?
   - Look for: "evaluated on", "tested on", "datasets", "benchmarks"
   - Extract dataset names (e.g., MNIST, ImageNet, GLUE)

**Important**:
- Only extract what is ACTUALLY MENTIONED in the paper
- If no experiments are described, say so
- Be specific with names
- If confidence is low, indicate it

**Output Format** (JSON):
```json
{{
  "baselines": [
    {{"name": "Baseline Name", "description": "Brief description from paper"}},
    ...
  ],
  "metrics": [
    {{"name": "Metric Name", "description": "What it measures"}},
    ...
  ],
  "datasets": [
    {{"name": "Dataset Name", "description": "Brief description"}},
    ...
  ],
  "has_experiments": true/false,
  "extraction_confidence": "high/medium/low",
  "notes": "Any relevant notes about the extraction"
}}
```

Extract the experimental setup now:
"""
        return prompt

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""

        # Try to extract JSON from response
        import re

        # Find JSON block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find any JSON-like content
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fallback: empty extraction
                return {
                    "baselines": [],
                    "metrics": [],
                    "datasets": [],
                    "has_experiments": False,
                    "extraction_confidence": "low",
                    "notes": "Could not parse response"
                }

        try:
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON: {e}")
            return {
                "baselines": [],
                "metrics": [],
                "datasets": [],
                "has_experiments": False,
                "extraction_confidence": "low",
                "notes": f"JSON parse error: {e}"
            }

    def extract_from_ground_truth_json(self, ground_truth_json: Dict) -> Dict[str, Any]:
        """
        Extract from already-redacted ground truth JSON.

        Args:
            ground_truth_json: Output from PaperProcessorTool

        Returns:
            Same format as extract_from_paper()
        """
        redacted = ground_truth_json.get("redacted_sections", {})

        # Combine all experimental text
        experimental_text = "\n\n".join([
            redacted.get("experiments_section", ""),
            redacted.get("results_section", ""),
            redacted.get("evaluation_section", ""),
            redacted.get("raw_experimental_content", "")
        ]).strip()

        if not experimental_text or len(experimental_text) < 50:
            return {
                "baselines": [],
                "metrics": [],
                "datasets": [],
                "has_experiments": False,
                "extraction_confidence": "low",
                "notes": "No experimental content found in ground truth JSON"
            }

        # Extract from experimental text
        return self.extract_from_paper(experimental_text)
