"""
Prompts for Experimental Setup Generation.

Centralizes all prompt templates used by agentic and non-agentic generators.
"""

from typing import Dict, Any, List
import json


def create_setup_prompt(paper_context: Dict[str, Any], focus_type: str = "balanced") -> str:
    """
    Create prompt for experimental setup generation with specific focus.
    
    Args:
        paper_context: Extracted paper context
        focus_type: Type of focus ("balanced", "diversity", "metrics", "datasets")
        
    Returns:
        Formatted prompt string
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

    # Focus instructions
    if focus_type == "balanced":
        focus = """**Focus for this setup**: Create a **balanced and comprehensive** experimental design that covers all aspects equally. Ensure completeness across baselines, metrics, datasets, and protocol.

**STEP 1 - SCAN RELATED WORK**: Before proposing any baselines, carefully read the Related Work section above. List out EVERY specific method/model name mentioned (e.g., "GraphCast", "Pangu-Weather", "BERT", "ResNet-50"). These are your primary baseline candidates.

**STEP 2 - USE EXACT NAMES**: When you find specific method names in Related Work, use those EXACT names as baselines. Do NOT generalize them (e.g., use "GraphCast" not "graph neural network model").

**STEP 3 - ADD DIVERSITY**: After including specific methods from Related Work, add 1-2 simple baselines for completeness."""

    elif focus_type == "diversity":
        focus = """**Focus for this setup**: Prioritize **baseline diversity**.

**CRITICAL - BASELINE EXTRACTION PROCESS**:
1. **READ Related Work section thoroughly** - It contains specific method names
2. **IDENTIFY at least 3-5 specific methods** mentioned by name (e.g., "Pangu-Weather", "GraphCast", "IFS")
3. **USE THOSE EXACT NAMES** as baselines - this is NON-NEGOTIABLE
4. **DO NOT use generic descriptions** like "numerical weather prediction models" - find the SPECIFIC model names
5. **Include varying complexity**: After specific methods from Related Work, add 1 simple baseline (e.g., "Persistence")

**Example of GOOD baselines**: "GraphCast", "Pangu-Weather", "ECMWF IFS", "Persistence Model"
**Example of BAD baselines**: "Graph neural network methods", "Transformer models", "Traditional NWP"""

    elif focus_type == "metrics":
        focus = """**Focus for this setup**: Prioritize **metric rigor and alignment**.

**CRITICAL - DOMAIN-SPECIFIC METRICS**:
1. **INFER from research question**: What specific aspect is being measured?
2. **USE domain-specific terminology**: Not "Accuracy" but "Forecast Accuracy", "Localization Accuracy", "Translation Accuracy"
3. **LOOK at the domain**: Weather forecasting uses RMSE/CRPS, NLP uses BLEU/perplexity, CV uses mAP/IoU
4. **INCLUDE both primary and diagnostic metrics**: What measures success + what provides insights

**Example**: For weather forecasting → "Forecast Accuracy", "RMSE", "CRPS" (NOT just "Accuracy", "Error Rate")"""

    elif focus_type == "datasets":
        focus = """**Focus for this setup**: Prioritize **dataset variety and appropriateness**.

**CRITICAL - DATASET IDENTIFICATION**:
1. **SCAN Introduction/Method** for mentions of data collection, dataset types, or benchmark names
2. **LOOK for specifics**: Dataset names ("WeatherBench 2"), data types ("RSSI fingerprints"), collection methods
3. **USE what's mentioned or implied** in paper context - don't hallucinate random datasets
4. **INCLUDE standard benchmarks** for this specific research domain if identified

**Example**: If paper mentions "WeatherBench 2" → include it. If method describes "indoor path data" → reference that."""
    
    else:
        # Default/Generic focus (used by non-agentic generator)
        focus = """**Your Task**: Design a complete experimental setup that includes:

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

6. **Overall Reasoning**: Explain why this experimental setup is appropriate for evaluating the proposed research."""

    guidelines = """
Design the experimental setup including:

1. **Hypotheses**: Formulate clear, testable hypotheses derived from the research question.
2. **Baselines**: Specific comparison methods (use exact names from Related Work when available).
3. **Control Groups**: Specific control conditions to isolate variables of interest.
4. **Metrics**: Domain-specific evaluation metrics aligned with the research question.
5. **Datasets**: Datasets mentioned or implied in the paper context.
6. **Methodological Rigor**:
   - Sample size justification (power analysis if applicable)
   - Randomization and blinding procedures
   - Significance tests
7. **Validity**:
   - Internal validity (causality)
   - External validity (generalizability)
   - Construct validity (measurement accuracy)
8. **Experimental Protocol**: Reproducible procedure with multiple runs, seeds, validation strategy.
9. **Reasoning**: Justify the overall setup design.

**Critical Requirements**:
- Use SPECIFIC names from Related Work, not generic descriptions
- Use domain-specific metric names that match the research area
- Reference datasets or data types mentioned in the paper context
- Be concrete and specific, never vague
- Address validity and rigor explicitly
"""

    # For non-agentic (generic focus), guidelines are already embedded in the focus text
    if focus_type not in ["balanced", "diversity", "metrics", "datasets"]:
        return base_context + focus + """

**Important Guidelines**:
- Design the setup to be rigorous, reproducible, and comprehensive
- Ensure baselines cover different complexity levels
- Choose metrics that align with the paper's research question
- Select datasets appropriate for the domain
- Specify concrete details (not vague descriptions)
- Justify your choices with clear rationale

Generate a complete experimental setup following these guidelines.
"""
    
    return base_context + focus + guidelines


def create_refinement_prompt(
    paper_context: Dict[str, Any],
    current_setup: Dict[str, Any],
    feedback: Dict[str, List[str]]
) -> str:
    """
    Create prompt to refine the best setup based on feedback.
    
    Args:
        paper_context: Extracted paper context
        current_setup: The setup to refine
        feedback: Feedback dictionary with weaknesses and suggestions
        
    Returns:
        Formatted prompt string
    """
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


def create_alignment_prompt(
    component_name: str,
    generated_list: List[str],
    ground_truth_list: List[str]
) -> str:
    """
    Create prompt for semantic alignment check.
    
    Args:
        component_name: Name of component (e.g., "Baselines", "Metrics")
        generated_list: List of generated items
        ground_truth_list: List of ground truth items
        
    Returns:
        Prompt string
    """
    gen_str = "\n".join([f"- {item}" for item in generated_list])
    gt_str = "\n".join([f"- {item}" for item in ground_truth_list])
    
    return f"""You are an expert research evaluator. Compare the following two lists of {component_name} to determine which ground truth items are covered by the generated list.

**Ground Truth {component_name}** (from the actual paper):
{gt_str}

**Generated {component_name}** (proposed by AI):
{gen_str}

**Task**:
For EACH item in the Ground Truth list, determine if it is effectively covered by ANY item in the Generated list.
- "Covered" means semantically equivalent or the generated item is a specific instance of the ground truth item.
- Be strict but fair. "Adam optimizer" covers "Adam". "Transformer" covers "BERT" (as a type), but "BERT" does not cover "Transformer" (too specific).

**Output Format**:
Return a JSON object with a single key "matches" containing a list of objects:
{{
  "matches": [
    {{
      "ground_truth_item": "Item Name",
      "covered": true/false,
      "generated_match": "Matching Item Name" (or null if no match),
      "reasoning": "Brief explanation"
    }},
    ...
  ]
}}
"""

