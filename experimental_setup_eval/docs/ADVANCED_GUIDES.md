# Advanced Guides

This document consolidates advanced usage guides for the experimental setup evaluation system.

---

## Table of Contents

1. [Manual Reproduction with ChatGPT/Claude](#manual-reproduction)
2. [Prompt Improvements](#prompt-improvements)
3. [Parser Improvements (2025-11-19)](#parser-improvements-2025-11-19)

---

## Manual Reproduction

**Goal**: Reproduce the experimental setup evaluation pipeline manually using ChatGPT/Claude without running code.

This shows exactly what to extract from a paper and what prompts to use to get the same results as the automated pipeline.

### The 3-Stage Manual Process

```
Paper PDF → Stage 1: Extract Context → Stage 2: Generate Setups → Stage 3: Compare to Ground Truth
```

### Stage 1: Paper Processing (Extract Context)

#### What to Extract

From the research paper, manually extract these sections:

| Section | What to Extract | Max Length | Where to Find |
|---------|----------------|------------|---------------|
| **Title** | Full paper title | N/A | First page, top |
| **Abstract** | Complete abstract | 1000 chars | After title, before intro |
| **Introduction** | Introduction section | 1500 chars | Section 1 or "Introduction" |
| **Related Work** | Related work/background | 1500 chars | Usually Section 2 |
| **Method** | Method description | 1500 chars | "Method" or "Methodology" |

#### What NOT to Read (Simulates Redaction)

**DO NOT read these sections**:
- ❌ Experiments
- ❌ Results
- ❌ Evaluation
- ❌ Implementation Details
- ❌ Experimental Setup

#### Template

Create a document with this structure:

```markdown
# Paper Context (Input for Stage 2)

**Title**: [Copy exact title]

**Abstract**: [Copy full abstract, up to 1000 characters]

**Domain**: [Identify: NLP, CV, ML-THEORY, OTHER-ML, RL, etc.]

**Introduction** (First 1500 characters):
[Copy introduction text...]

**Related Work** (First 1500 characters):
[Copy related work section...]

**Method Description** (First 1500 characters):
[Copy method/approach section...]
```

### Stage 2A: Extract Research Question

#### Prompt for LLM

```
You are an expert researcher. Given this paper's context, extract the main research question or hypothesis.

# Paper Title
[Paste title]

# Paper Context
[Paste introduction + abstract, max 2000 characters]

# Task
Identify and state the main research question or hypothesis in 1-2 sentences. Focus on what the paper is trying to answer or prove.

Response format: Just the research question/hypothesis, nothing else.
```

### Stage 2B: Generate Experimental Setup (Non-Agentic)

#### Full Prompt Template

```
You are an expert in machine learning research methodology. Your task is to design a comprehensive experimental setup for evaluating the research proposed in the following paper.

**Paper Title**: [PASTE TITLE]

**Abstract**: [PASTE ABSTRACT]

**Research Question**: [PASTE RESEARCH QUESTION FROM STAGE 2A]

**Domain**: [PASTE DOMAIN]

**Introduction**: [PASTE FIRST 1000 CHARS OF INTRODUCTION]

**Related Work**: [PASTE FIRST 1500 CHARS OF RELATED WORK]

**Method Description**: [PASTE FIRST 1500 CHARS OF METHOD]

---

**Your Task**: Design a comprehensive experimental setup.

**STEP 1 - SCAN RELATED WORK**: Before proposing any baselines, carefully read the Related Work section above. List out EVERY specific method/model name mentioned (e.g., "GraphCast", "Pangu-Weather", "BERT"). These are your primary baseline candidates.

**STEP 2 - USE EXACT NAMES**: When you find specific method names in Related Work, use those EXACT names as baselines. Do NOT generalize them.

**STEP 3 - ADD DIVERSITY**: After including specific methods from Related Work, add 1-2 simple baselines for completeness.

**Design the setup including**:

1. **Hypotheses**: Formulate clear, testable hypotheses derived from the research question.
2. **Baselines**: Specific comparison methods (as identified in Step 1 & 2).
3. **Control Groups**: Specific control conditions to isolate variables of interest.
4. **Metrics**: Domain-specific evaluation metrics aligned with the research question (e.g., "Localization Accuracy" for indoor localization).
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

Generate a complete experimental setup following these guidelines.

**Output Format**: Provide a structured response with these sections:
- Baselines (list each with name, description, complexity level, rationale)
- Metrics (list each with name, description, type, higher_is_better, expected_range, rationale)
- Datasets (list each with name, description, size, domain, split, rationale)
- Significance Tests (list each with test_name, when_to_use, assumptions)
- Experimental Protocol (num_runs, random_seeds, validation_strategy, hyperparameter_tuning, compute_budget)
- Overall Reasoning (paragraph explaining the setup)
```

Save the LLM's response as **"AI_Generated_Setup.txt"**

### Stage 2C: Generate Agentic Setup (Optional)

For more thorough evaluation, generate 3 different setups with varied focus:

#### Draft 0: Balanced (Recommended)
Use the prompt from Stage 2B above. This focuses on completeness and balance across all aspects.

#### Draft 1: Diversity-Focused
Add this focus instruction:
"**Focus for this setup**: Prioritize **baseline diversity**. **CRITICAL**: Identify at least 3-5 specific methods mentioned by name in Related Work. Do NOT use generic descriptions."

#### Draft 2: Metric-Focused
Add this focus instruction:
"**Focus for this setup**: Prioritize **metric rigor and alignment**. **CRITICAL**: Infer domain-specific metrics from the research question. Use specific terminology (e.g., 'Forecast Accuracy' not just 'Accuracy')."

#### Draft 3: Dataset-Focused (Optional)
Add this focus instruction:
"**Focus for this setup**: Prioritize **dataset variety**. **CRITICAL**: Scan Introduction/Method for specific dataset names or data types."

**Then**: Review all 3 drafts and select the best one (or combine elements).

### Stage 3: Extract Ground Truth

Read the **Experiments/Results** section (which you skipped earlier).

#### Create Ground Truth File

**"Ground_Truth.txt"**:

```markdown
# Ground Truth Experimental Setup

## Baselines
[List all baseline methods from paper, use exact names]
- [Baseline 1 name]: [Brief description]
- [Baseline 2 name]: [Brief description]
...

## Metrics
[List all metrics from paper]
- [Metric 1 name]: [What it measures]
- [Metric 2 name]: [What it measures]
...

## Datasets
[List all datasets from paper]
- [Dataset 1 name]: [Description]
- [Dataset 2 name]: [Description]
...
```

### Stage 4: Calculate Alignment

### Stage 4: Calculate Alignment

#### Quantitative Scoring
For each category, calculate F1 Score (Harmonic mean of Precision and Recall):

1. **Baselines/Metrics/Datasets F1**:
   - Recall = (# AI items matching GT) / (# total GT items)
   - Precision = (# AI items matching GT) / (# total AI items)
   - F1 = 2 * (Precision * Recall) / (Precision + Recall)

#### Qualitative Alignment
Manually check alignment for:
- **Hypotheses**: Do AI hypotheses match the core testable claims in GT?
- **Controls**: Are control groups semantically equivalent?
- **Protocol**: Are key elements (runs, seeds, cross-validation) present in both?

#### Overall Score
Weighted average (automated system uses):
- Quantitative (Baselines, Metrics, Datasets)
- Qualitative (Hypotheses, Controls, Protocol)

### Expected Results

Based on analysis:

| Component | Typical Alignment | Why |
|-----------|------------------|-----|
| Baselines | 0-50% | AI can't predict which specific methods authors chose |
| Metrics | 50-75% | AI can infer some domain metrics from context |
| Datasets | 0-25% | Authors often use custom datasets not in intro |

**Note**: Perfect alignment (100%) is unlikely because related work mentions many methods but authors choose a subset.

### Tips for Better Results

**DO**:
- ✅ Copy full sections up to character limits
- ✅ Include all method names from related work
- ✅ Preserve technical terminology

**DON'T**:
- ❌ Summarize or paraphrase
- ❌ Skip related work section
- ❌ Read experiments before generating setup

---

## Prompt Improvements

### Problem Analysis

After analyzing 0% alignment results, we identified critical weaknesses:

#### Root Causes

1. **Missing Related Work Context**
   - AI had NO information about prior methods
   - Prompts asked for "state-of-the-art" without providing any
   - Result: AI invented plausible but wrong baselines

2. **Generic Instructions**
   - No emphasis on specificity
   - No guidance to extract names from paper
   - Result: Generic metrics instead of domain-specific ones

3. **No Dataset Guidance**
   - Didn't tell AI to look for data collection methods
   - Result: Standard benchmarks instead of paper-specific datasets

4. **Poor Context Extraction**
   - Title extraction failing
   - Abstract often empty
   - Related work not passed to generators

### Improvements Made

#### 1. Added Related Work & Introduction

**Before:**
```python
title = paper_context.get("title", "Unknown Title")
abstract = paper_context.get("abstract", "")
research_question = paper_context.get("research_question", "")
```

**After:**
```python
related_work = paper_context.get("related_work", "")  # NEW
introduction = paper_context.get("introduction", "")   # NEW
```

#### 2. Explicit Name Extraction Instructions

Added **CRITICAL** markers and examples:

```
**CRITICAL**: Look carefully at the Related Work section to identify specific prior methods by name.

**Example**: If Related Work mentions "BERT" and "GPT-2", include those as baselines, not generic "transformer model".
```

#### 3. Domain-Specific Guidance

```
**CRITICAL**: Look at the research question and domain to infer metrics.

**Example**: For indoor localization, use "Localization Accuracy" not just "Accuracy".
```

#### 4. Dataset Context

```
**CRITICAL**: If the Introduction or Method mentions data collection, reference those.

Include datasets implied by the paper (e.g., "indoor paths", "RSSI fingerprints").
```

### Expected Impact

**Before:**
- Baselines: 0/4 recall
- Metrics: 50% recall
- Datasets: 0/2 recall

**After (Expected):**
- Baselines: 50-75% recall
- Metrics: 75-100% recall
- Datasets: 25-50% recall

### Recent Improvements (Dec 2025)

#### 1. Focus-Based Prompting
Moved from a single monolithic prompt to varied "Focus Types" for agentic generation:
- **Balanced**: Combines all best practices (Used for Draft 0)
- **Diversity**: Forces extensive Related Work scanning
- **Metrics**: Forces domain-specific terminology
- **Datasets**: Forces detailed infrastructure inference

#### 2. Structured Guidelines
Prompts now explicitly request:
- **Testable Hypotheses**: Not just general questions
- **Control Groups**: Explicit separation from baselines
- **Methodological Rigor**: Power analysis, blinding, randomization
- **Validity awareness**: Internal/External/Construct validity checks

#### 3. Expanded Context
- **Introduction**: Increased to 3000 chars
- **Related Work**: Increased to 4000 chars (Critical for baseline extraction)

---

## Parser Improvements (2025-11-19)

### What Changed

 Enhanced **PaperProcessorTool** with better HTML handling and section extraction.

### Improvements Made

1. **HTML Tag Cleaning**
   - Removes `<figure>`, `<embed>`, `<figcaption>` tags common in PDF conversions
   - Converts HTML headers (`<h1>`, `<h2>`) to markdown (`#`, `##`)
   - Cleans LaTeX/PDF artifacts (`data-latex-*`, `id` attributes)
   - Normalizes whitespace

2. **Smart Title Extraction**
   - Skips date patterns (e.g., `_2024-10-23_`)
   - Filters out metadata and short fragments  
   - Looks for first substantial text with markdown formatting
   - Falls back to section names if needed

3. **Better Section Detection**
   - Uses regex-based header matching (`^#{1,4}\s+`)
   - Handles both markdown headers and bold section names
   - More robust parsing of mixed formats

4. **Abstract Fallback Logic**
   - When no explicit abstract section exists
   - Extracts from header section after title
   - Skips dates and finds first substantial paragraph

5. **Increased Context Limits**
   - Introduction: 1000 → 3000 chars
   - Related Work: 1500 → 4000 chars
   - Method: 1500 → 3000 chars
   - Preserves more baseline/dataset names

6. **Additional Context Fields**
   - Now passes `introduction` to generators
   - Now passes `related_work` to generators
   - Critical for baseline name extraction

### Code Changes

**File**: [`paper_processor.py`](../experimental_setup_eval/tools/paper_processor.py)

```python
def _clean_markdown(self, content: str) -> str:
    """Remove HTML tags and normalize formatting."""
    # Remove HTML figure tags
    content = re.sub(r'<figure[^>]*>.*?</figure>', '', content, flags=re.DOTALL)
    content = re.sub(r'<embed[^>]*>', '', content)
    # Convert HTML headers to markdown
    content = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1', content, flags=re.DOTALL)
    # ... (clean LaTeX artifacts, normalize whitespace)
    return content
```

### Impact on Results

**Before** (with HTML tag issues):
- Title: "_2024-10-23_" ❌
- Abstract: Empty string ❌  
- Sections: Broken/fragmented ❌
- Alignment: 0% ❌

**After** (with improvements):
- Title: Correct paper title ✅
- Abstract: Properly extracted ✅
- Sections: Clean markdown ✅
- Expected Alignment: 40-80%+ ✅

### Testing

Run validation to see improvements:

```bash
python experimental_setup_eval/validate_and_filter_papers.py data/papers/{collection}/
```

Expected: Higher validation pass rates, better title extraction, more complete section detection.

---

## Summary

These advanced guides provide:

1. **Manual Reproduction**: Step-by-step manual process for understanding the pipeline
2. **Prompt Improvements**: How prompts were improved to increase alignment
3. **Marker Integration**: Better PDF conversion for higher quality extraction

Use these when you need to:
- Understand pipeline internals deeply
- Debug alignment issues
- Improve text extraction quality
- Reproduce results without code
