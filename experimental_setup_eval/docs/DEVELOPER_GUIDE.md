# Developer Guide

**Architecture and extension details for developers working on the experimental setup evaluation system.**

Last Updated: 2025-11-19

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Extension Design](#extension-design)
3. [Key Components](#key-components)
4. [Integration with AI Scientist v2](#integration-with-ai-scientist-v2)
5. [Adding New Features](#adding-new-features)

---

## Architecture Overview

### High-Level System Design

```
AI Scientist v2 Core (UNCHANGED)
        ↑ uses
        |
experimental_setup_eval/ (NEW, peer-level extension)
```

**Design Principle**: **Minimal Modification, Maximum Reuse**

- No modifications to AI Scientist core
- Peer-level extension in separate directory
- Follows established patterns (BaseTool, FunctionSpec)
- Can evolve independently

### Project Structure

```
AI-Scientist-v2-base/
├── ai_scientist/                    # AI Scientist v2 core (UNCHANGED)
│   ├── treesearch/backend/
│   │   ├── query() function
│   │   └── utils/FunctionSpec
│   ├── tools/base_tool.py          # BaseTool abstract class
│   └── llm.py                      # create_client(), get_response_from_llm()
│
├── experimental_setup_eval/        # Our extension (peer-level)
│   ├── tools/
│   │   ├── paper_processor.py     # Parse & redact papers
│   │   ├── ground_truth_extractor.py
│   │   └── suitability_checker.py
│   ├── generators/
│   │   ├── non_agentic_generator.py
│   │   └── agentic_generator.py
│   ├── evaluation/
│   │   ├── quality_scorer.py
│   │   ├── comparator.py
│   │   └── alignment_scorer.py
│   ├── data_collection/
│   │   ├── collect_papers.py
│   │   └── improved_paper_extraction.py
│   ├── run_pipeline.py            # Main 3-stage pipeline
│   └── validate_and_filter_papers.py
│
├── data/papers/                   # Collected papers (immutable)
├── runs/                          # Pipeline results
└── temp/                          # Documentation
```

---

## Extension Design

### Design Principles

From the original [EXTENSION.md](archive/EXTENSION.md) design document:

1. **Reuse Existing Abstractions**: BaseTool, LLM integration, backend system
2. **Modular Design**: New components don't modify core
3. **Configurable Evaluation**: Easy to adjust criteria and weights
4. **Two-Pipeline Architecture**: Separation of generation and evaluation
5. **Extensible Framework**: Easy to add new features
1.  **Reuse Existing Abstractions**: BaseTool, LLM integration, backend system
2.  **Modular Design**: New components don't modify core
3.  **Configurable Evaluation**: Easy to adjust criteria and weights
4.  **Two-Pipeline Architecture**: Separation of generation and evaluation
5.  **Extensible Framework**: Easy to add new features

### Key Dependencies

-   **AI Scientist v2**: Core framework
-   **LLM APIs**: OpenAI (GPT-4o) or Anthropic (Claude Sonnet)
-   **PDF Processing**: `pymupdf4llm` for PDF→markdown conversion
-   **Paper Collection**: Semantic Scholar API
-   **Data Processing**: `pandas`, `numpy` for analysis

**Recent Improvements (2025-11-19)**:
-   Enhanced HTML tag cleaning in parser
-   Better title and section extraction
-   Increased context preservation (3000-4000 chars vs 1500)
-   Additional context fields (introduction, related_work)

### What We Reuse ✓

From AI Scientist v2:
-   **BaseTool Pattern**: Abstract base class for tools
-   **FunctionSpec**: Structured outputs with JSON schema validation
-   **LLM Integration**: `create_client()`, `get_response_from_llm()`, `query()`
-   **Backend Abstraction**: Unified interface across LLM providers
-   **PDF Processing**: `pymupdf4llm` for markdown conversion

### What We Extend

New components:
-   **Tools**: `PaperProcessorTool`, `GroundTruthExtractor`
-   **Generators**: `NonAgenticSetupGenerator`, `AgenticSetupGenerator`
-   **Evaluators**: `SetupQualityScorer`, `AlignmentScorer`, `SetupComparator`
-   **Workflows**: Paper collection, validation, 3-stage pipeline

---

## Key Components

### 1. PaperProcessorTool

**File**: [experimental_setup_eval/tools/paper_processor.py](../experimental_setup_eval/tools/paper_processor.py)

**Purpose**: Extract and process paper content with redaction strategy.

**Key Methods**:
```python
def process_paper(self, paper_path: str) -> Dict:
    """
    Parse paper markdown and extract sections.

    Returns:
        {
            "title": str,
            "abstract": str,
            "introduction": str,
            "related_work": str,
            "method_description": str,
            "research_question": str,
            "domain": str,
            "redacted_sections": {
                "experiments": str,
                "results": str,
                "evaluation": str
            }
        }
    """
```

**Design Notes**:
- Uses LLM to extract research question from context
- Identifies domain (NLP, CV, RL, etc.) for domain-specific processing
- **Critical**: AI generators never see redacted sections
- Section detection uses both markdown headers and LLM parsing

### 2. Setup Generators

#### NonAgenticSetupGenerator

**File**: [experimental_setup_eval/generators/non_agentic_generator.py](../experimental_setup_eval/generators/non_agentic_generator.py)

**Strategy**: Single LLM call with comprehensive prompt

**Key Method**:
```python
def generate_setup(self, paper_context: Dict, domain: str) -> Dict:
    """
    Generate experimental setup from paper context.

    Returns:
        {
            "baselines": [{"name": str, "description": str, "rationale": str}],
            "metrics": [{"name": str, "type": str, "rationale": str}],
            "datasets": [{"name": str, "description": str, "rationale": str}],
            "significance_tests": [...],
            "experimental_protocol": {...},
            "reasoning": str
        }
    """
```

**Prompt Strategy**:
- Includes introduction, related work, method context
- **CRITICAL** markers emphasize name extraction from related work
- Domain-specific examples (NLP vs CV vs RL)
- Structured output via FunctionSpec

#### AgenticSetupGenerator

**File**: [experimental_setup_eval/generators/agentic_generator.py](../experimental_setup_eval/generators/agentic_generator.py)

**Strategy**: Multi-draft generation with selection

**Process**:
1. Generate N drafts (default 3) with varied focus:
   - Draft 0: Balanced
   - Draft 1: Baseline-focused
   - Draft 2: Metric-focused
   - Draft 3: Dataset-focused (if N=4)

2. Score each draft with SetupQualityScorer

3. Select best draft based on scores

**Key Method**:
```python
def generate_setup(self, paper_context: Dict, domain: str, num_drafts: int = 3) -> Dict:
    """
    Generate multiple setup drafts and select best.

    Returns: Best setup (same format as NonAgenticSetupGenerator)
    """
```

### 3. Evaluation Components

#### SetupQualityScorer

**File**: [experimental_setup_eval/evaluation/quality_scorer.py](../experimental_setup_eval/evaluation/quality_scorer.py)

**Purpose**: Score setup quality before seeing ground truth

**Scoring Dimensions** (0-1 scale):
```python
{
    "completeness": 0.3,      # Has all required components?
    "diversity": 0.2,         # Baselines span complexity?
    "alignment": 0.25,        # Metrics match domain?
    "datasets": 0.15,         # Well-specified datasets?
    "protocol": 0.1           # Rigorous protocol?
}
```

**Total Score**: Weighted average of dimensions

#### AlignmentScorer

**File**: [experimental_setup_eval/evaluation/alignment_scorer.py](../experimental_setup_eval/evaluation/alignment_scorer.py)

**Purpose**: Compare AI-generated setup with ground truth

**Matching Strategy**: Fuzzy string matching
```python
def fuzzy_match(ai_name: str, gt_name: str) -> bool:
    """
    Returns True if:
    - Exact match
    - Substring match
    - High similarity (>0.8 using difflib)
    """
```

**Scores**:
- **Baseline Recall**: % of true baselines found
- **Metric Recall**: % of true metrics found
- **Dataset Recall**: % of true datasets found
- **Overall**: Weighted average (40% baselines, 40% metrics, 20% datasets)

#### SetupComparator

**File**: [experimental_setup_eval/evaluation/comparator.py](../experimental_setup_eval/evaluation/comparator.py)

**Purpose**: Compare non-agentic vs agentic setups

**Comparison Dimensions**:
- Quality score differences
- Time and cost analysis
- Determines winner based on quality, cost, and alignment

### 4. Ground Truth Extractor

**File**: [experimental_setup_eval/tools/ground_truth_extractor.py](../experimental_setup_eval/tools/ground_truth_extractor.py)

**Purpose**: Extract actual experimental setup from paper

**Method**:
```python
def extract_ground_truth(self, paper_markdown: str) -> Dict:
    """
    Extract actual baselines, metrics, datasets from experiments section.

    Uses LLM to parse experimental sections and identify:
    - Baseline methods compared
    - Metrics used for evaluation
    - Datasets tested on

    Returns:
        {
            "baselines": [str],
            "metrics": [str],
            "datasets": [str]
        }
    """
```

---

## Integration with AI Scientist v2

### Leveraging Existing Infrastructure

#### 1. LLM Integration

**From**: [ai_scientist/llm.py](../ai_scientist/llm.py)

```python
from ai_scientist.llm import create_client, get_response_from_llm

# In generators and extractors
client = create_client(self.model)
response = get_response_from_llm(
    messages=[{"role": "user", "content": prompt}],
    client=client,
    model=self.model,
    system_message=system_prompt,
    temperature=0.7
)
```

#### 2. Structured Outputs

**From**: [ai_scientist/treesearch/backend/utils.py](../ai_scientist/treesearch/backend/utils.py)

```python
from ai_scientist.treesearch.backend.utils import FunctionSpec, query

# Define schema
experimental_setup_spec = FunctionSpec(
    name="generate_experimental_setup",
    description="Generate experimental setup",
    parameters={
        "type": "object",
        "properties": {
            "baselines": {...},
            "metrics": {...},
            "datasets": {...}
        },
        "required": ["baselines", "metrics", "datasets"]
    }
)

# Use with query
result = query(
    messages=[...],
    model="claude-3-5-sonnet-20241022",
    func_spec=experimental_setup_spec
)
```

#### 3. Tool Framework

**From**: [ai_scientist/tools/base_tool.py](../ai_scientist/tools/base_tool.py)

```python
from ai_scientist.tools.base_tool import BaseTool

class PaperProcessorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="paper_processor",
            description="Process and redact research papers",
            parameters={...}
        )

    def use_tool(self, **kwargs):
        # Implementation
        pass
```

---

## Adding New Features

### Adding a New Evaluation Metric

1. **Define metric in quality_scorer.py**:
```python
def score_new_dimension(self, setup: Dict) -> float:
    """
    Score a new quality dimension.

    Returns: Score between 0 and 1
    """
    # Implementation
    pass
```

2. **Add to scoring weights**:
```python
WEIGHTS = {
    "completeness": 0.25,     # Reduced
    "diversity": 0.2,
    "alignment": 0.2,         # Reduced
    "datasets": 0.15,
    "protocol": 0.1,
    "new_dimension": 0.1      # NEW
}
```

3. **Update in calculate_quality_score()**:
```python
scores["new_dimension"] = self.score_new_dimension(setup)
```

### Adding a New Generator

1. **Create new generator file**:
```python
# experimental_setup_eval/generators/my_generator.py

from ai_scientist.llm import create_client, get_response_from_llm

class MySetupGenerator:
    def __init__(self, model="gpt-4o-2024-11-20"):
        self.model = model
        self.client = create_client(model)

    def generate_setup(self, paper_context: Dict, domain: str) -> Dict:
        # Your generation logic
        pass
```

2. **Integrate in run_pipeline.py**:
```python
from experimental_setup_eval.generators.my_generator import MySetupGenerator

# In main pipeline
my_gen = MySetupGenerator()
my_setup = my_gen.generate_setup(paper_context, domain)

# Save and evaluate
save_json(my_setup, f"{output_dir}/my_setup.json")
```

3. **Add to comparison**:
```python
# Compare with existing approaches
my_alignment = alignment_scorer.score_alignment(my_setup, ground_truth)
```

### Adding a New Tool

1. **Inherit from BaseTool**:
```python
# experimental_setup_eval/tools/my_tool.py

from ai_scientist.tools.base_tool import BaseTool

class MyTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="What this tool does",
            parameters={"param1": "description"}
        )

    def use_tool(self, param1):
        # Implementation
        result = do_something(param1)
        return result
```

2. **Use in pipeline**:
```python
from experimental_setup_eval.tools.my_tool import MyTool

tool = MyTool()
result = tool.use_tool(param1="value")
```

### Modifying Prompts

**Location**: Prompts are in generator files

**Non-Agentic**: [non_agentic_generator.py](../experimental_setup_eval/generators/non_agentic_generator.py) → `_create_prompt()`

**Agentic**: [agentic_generator.py](../experimental_setup_eval/generators/agentic_generator.py) → `_create_varied_prompt()`

**Best Practices**:
- Use **CRITICAL** markers for important instructions
- Provide domain-specific examples
- Include related work context for baseline extraction
- Request structured output format

**Example Addition**:
```python
def _create_prompt(self, paper_context: Dict, domain: str) -> str:
    # Existing prompt
    prompt = f"""
    ...existing content...

    # NEW SECTION
    **CRITICAL**: Also consider the following when designing the setup:
    - [Your new instruction]
    - [Example]

    ...rest of prompt...
    """
    return prompt
```

---

## Testing

### Unit Tests

Create tests in `tests/` directory:

```python
# tests/test_generators.py

import pytest
from experimental_setup_eval.generators.non_agentic_generator import NonAgenticSetupGenerator

def test_generator():
    """Test setup generation."""
    gen = NonAgenticSetupGenerator()

    paper_context = {
        "title": "Test Paper",
        "abstract": "...",
        "research_question": "...",
        "domain": "NLP"
    }

    setup = gen.generate_setup(paper_context, "NLP")

    assert "baselines" in setup
    assert len(setup["baselines"]) > 0
    assert "metrics" in setup
```

### Integration Tests

Test full pipeline:

```python
# tests/test_pipeline.py

def test_full_pipeline():
    """Test end-to-end pipeline."""
    # Process paper
    processor = PaperProcessorTool()
    context = processor.process_paper("test_paper.md")

    # Generate setup
    gen = NonAgenticSetupGenerator()
    setup = gen.generate_setup(context, context["domain"])

    # Extract ground truth
    extractor = GroundTruthExtractor()
    gt = extractor.extract_ground_truth("test_paper.md")

    # Score alignment
    scorer = AlignmentScorer()
    alignment = scorer.score_alignment(setup, gt)

    assert alignment["overall"] >= 0.0
    assert alignment["overall"] <= 1.0
```

---

## Performance Optimization

### Reducing Costs

1. **Use cheaper models for testing**:
```python
# In generator __init__
self.model = "gpt-4o-mini"  # Instead of gpt-4o or claude
```

2. **Reduce agentic drafts**:
```python
# In run_pipeline.py
agentic_gen = AgenticSetupGenerator(num_drafts=2)  # Instead of 3-4
```

3. **Cache intermediate results**:
```python
# Add caching in run_pipeline.py
if os.path.exists(f"{output_dir}/paper_context.json"):
    context = load_json(f"{output_dir}/paper_context.json")
else:
    context = processor.process_paper(paper_path)
    save_json(context, f"{output_dir}/paper_context.json")
```

### Improving Speed

1. **Parallel processing** (multiple papers):
```python
from multiprocessing import Pool

def process_paper_wrapper(args):
    paper_path, output_dir = args
    # Process single paper
    return result

with Pool(4) as pool:
    results = pool.map(process_paper_wrapper, paper_args)
```

2. **Async LLM calls** (within single paper):
```python
import asyncio

async def generate_all_drafts(paper_context, domain):
    tasks = [
        generate_draft_async(paper_context, domain, i)
        for i in range(num_drafts)
    ]
    return await asyncio.gather(*tasks)
```

---

## Code Style and Conventions

### Import Organization

```python
# Standard library
import os
import json
from pathlib import Path

# Third-party
import numpy as np
from tqdm import tqdm

# AI Scientist v2
from ai_scientist.llm import create_client, get_response_from_llm
from ai_scientist.tools.base_tool import BaseTool

# Local imports
from experimental_setup_eval.tools.paper_processor import PaperProcessorTool
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `PaperProcessorTool`)
- **Functions**: snake_case (e.g., `generate_setup`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_MODEL`)
- **Private methods**: _leading_underscore (e.g., `_create_prompt`)

### Documentation

Use docstrings with clear descriptions:

```python
def generate_setup(self, paper_context: Dict, domain: str) -> Dict:
    """
    Generate experimental setup from paper context.

    Args:
        paper_context: Dictionary with title, abstract, research_question, etc.
        domain: Research domain (NLP, CV, RL, etc.)

    Returns:
        Dictionary with baselines, metrics, datasets, protocol, reasoning

    Raises:
        ValueError: If paper_context missing required fields
    """
```

---

## Summary

**Extension Architecture**:
- Peer-level to AI Scientist v2
- Reuses existing patterns and tools
- No modifications to core
- Fully independent evolution

**Key Components**:
- **Tools**: Paper processing, ground truth extraction
- **Generators**: Non-agentic, agentic, (extensible)
- **Evaluators**: Quality scoring, alignment scoring, comparison

**Integration Points**:
- LLM integration via `create_client()`, `get_response_from_llm()`
- Structured outputs via `FunctionSpec` and `query()`
- Tool framework via `BaseTool`

**Extensibility**:
- Easy to add new generators
- Easy to add new evaluation metrics
- Easy to modify prompts
- Clear separation of concerns

For user-facing documentation, see [USER_GUIDE.md](USER_GUIDE.md).
