# Experimental Setup Evaluation - Complete User Guide

**Complete system for evaluating AI-generated experimental setups against human choices.**

Last Updated: 2025-11-19

---

## Table of Contents

1. [What This Does](#what-this-does)
2. [Quick Start](#quick-start)
3. [Paper Collection](#paper-collection)
4. [Paper Validation](#paper-validation)
5. [Running the Pipeline](#running-the-pipeline)
6. [Viewing Results](#viewing-results)
7. [Troubleshooting](#troubleshooting)

---

## What This Does

Evaluates whether AI can design experimental setups as well as humans by:

1. **Extracting research context** from papers (without seeing experiments)
2. **Generating experimental setups** using non-agentic and agentic approaches
3. **Comparing with ground truth** (what authors actually did)

**Research Question**: Can AI predict the experimental choices human researchers make?

---

## Quick Start

### Minimal Setup (3 Commands)

```bash
# 1. Collect papers (creates papers_to_process.txt automatically)
python experimental_setup_eval/data_collection/collect_papers.py --query "transformer" --max-results 2 --download-pdfs --convert-markdown

# 2. Set API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# 3. Run full pipeline
python experimental_setup_eval/run_pipeline.py papers_to_process.txt
```

**Results**: `runs/{timestamp}_full_eval/` with complete evaluation

### Recommended Workflow (With Validation)

```bash
# 1. Collect papers
python experimental_setup_eval/data_collection/collect_papers.py --query "transformer" --max-results 10 --download-pdfs --convert-markdown

# 2. Validate papers (filters out poor quality)
python experimental_setup_eval/validate_and_filter_papers.py data/papers/{timestamp}_collection/

# 3. Run pipeline on validated papers only
python experimental_setup_eval/run_pipeline.py data/papers/{timestamp}_collection/validated_papers_to_process.txt
```

---

## Paper Collection

### Basic Collection

Collect papers from Semantic Scholar API:

```bash
python experimental_setup_eval/data_collection/collect_papers.py \
  --query "transformer neural networks" \
  --max-results 10 \
  --download-pdfs \
  --convert-markdown
```

**Output**: `data/papers/{timestamp}_collection/` containing:
- Each paper in its own folder
- `paper.pdf` - Original PDF
- `paper.md` - Markdown conversion (from PDF)
- `paper_clean.md` - Improved markdown (from LaTeX if available)
- `metadata.json` - Paper metadata
- `papers_to_process.txt` - List of markdown files ready for pipeline

### Advanced Collection Options

```bash
python experimental_setup_eval/data_collection/collect_papers.py \
    --query "weather forecasting" \
    --max-results 20 \
    --min-citations 10 \
    --year-range 2020-2024 \
    --fields-of-study "Computer Science" \
    --download-pdfs \
    --convert-markdown
```

**Parameters**:
- `--query`: Search query
- `--max-results`: Number of papers to collect (default: 10)
- `--min-citations`: Minimum citation count (default: 0)
- `--year-range`: Publication year range (e.g., "2020-2024")
- `--fields-of-study`: Filter by field (e.g., "Computer Science", "Physics")
- `--venue`: Filter by venue (e.g., "NeurIPS", "ICML")
- `--download-pdfs`: Download PDF files
- `--convert-markdown`: Convert PDFs to markdown

### Collection Output Structure

```
data/papers/{timestamp}_collection/
├── collection_summary.json          # Collection metadata
├── papers_to_process.txt            # Ready-to-use file list
├── Paper_Title_1_{id}/
│   ├── paper.pdf
│   ├── paper.md                     # PDF → Markdown
│   ├── paper_clean.md               # LaTeX → Markdown (preferred)
│   ├── latex_source/                # If from ArXiv
│   └── metadata.json
└── Paper_Title_2_{id}/
    └── ...
```

### PDF Conversion Details

The collection script uses **two conversion strategies**:

1. **LaTeX → Markdown** (Preferred)
   - For ArXiv papers with LaTeX source
   - Uses `pandoc` for high-quality conversion
   - Preserves structure, equations, tables
   - Output: `paper_clean.md`

2. **PDF → Markdown** (Fallback)
   - For papers without LaTeX source
   - Uses `pymupdf4llm` for extraction
   - Output: `paper.md`

**Pipeline automatically prefers** `paper_clean.md` when available.

### Parser Improvements (2025-11-19)

The **PaperProcessorTool** has been enhanced to handle common PDF-to-markdown issues:

- **HTML tag cleaning**: Removes `<figure>`, `<embed>`, and other HTML artifacts common in PDF conversions
- **Smart title extraction**: Skips dates (e.g., `_2024-10-23_`) and metadata to find actual paper titles
- **Better section detection**: Uses regex-based header matching for more robust parsing
- **Abstract fallback**: Extracts abstract from header section when no explicit abstract section exists
- **Increased context limits**: Parser now passes more content (3000-4000 chars vs 1000-1500) to preserve specific baseline/dataset names

---

## Paper Validation

**Why validate?** Ensures only papers with complete extraction are processed, preventing wasted LLM calls and improving results.

### Basic Validation

```bash
python experimental_setup_eval/validate_and_filter_papers.py \
    data/papers/{timestamp}_collection/
```

**Output**:
- `validated_papers_to_process.txt` - Filtered list of quality papers
- `validation_report.json` - Detailed validation results

### Validation Criteria

Papers must pass these checks:

**Critical (Must Pass)**:
- ✓ Markdown file exists and readable
- ✓ File size ≥ 1000 bytes
- ✓ Valid title (not "Unknown Title" or date)
- ✓ Abstract ≥ 100 characters
- ✓ Introduction ≥ 300 characters
- ✓ Method description ≥ 200 characters
- ✓ Total content ≥ 1000 characters

**Warnings (Non-blocking)**:
- ⚠ Missing related work (common, often in introduction)
- ⚠ Title mismatch with metadata

### Custom Validation Thresholds

```bash
python experimental_setup_eval/validate_and_filter_papers.py \
    data/papers/{timestamp}_collection/ \
    --min-title-length 15 \
    --min-abstract-length 200 \
    --min-intro-length 500 \
    --min-method-length 300
```

### Validation Report

View detailed validation results:

```bash
cat data/papers/{timestamp}_collection/validation_report.json
```

Example report:
```json
{
  "summary": {
    "total_papers": 10,
    "passed": 7,
    "failed": 3,
    "pass_rate": 70.0,
    "average_score": 82.5
  },
  "results": [
    {
      "paper_id": "...",
      "passed": true,
      "score": 90.0,
      "issues": [],
      "warnings": ["Related work section missing"]
    }
  ]
}
```

---

## Running the Pipeline

### The 3-Stage Pipeline

#### Stage 1: Paper Processing
- **Tool**: PaperProcessorTool (with HTML cleaning and improved parsing)
- **Extracts**: title, abstract, **introduction**, **related work**, research question, domain, method
- **Redacts**: experimental sections (AI never sees them)
- **Key Improvement**: Now passes introduction and related work sections which contain specific baseline method names
- **Output**: `paper_context.json` (AI input), stores ground truth for later

#### Stage 2: Setup Generation (Dual-Mode)

**Non-Agentic**:
- Single LLM call
- Fast & cheap (~$0.05, 12s per paper)
- Proposes: baselines, metrics, datasets, protocol

**Agentic**:
- Multi-draft approach (3 drafts with different focus)
- Selects best using quality scorer
- Slower but more thorough (~$0.20, 60s per paper)

**Comparison**:
- Quality scores (0-1): completeness, diversity, alignment
- Cost-benefit analysis

#### Stage 3: Ground Truth Comparison

**Ground Truth Extraction**:
- Extracts actual baselines/metrics/datasets from experimental sections
- Uses LLM to parse

**Alignment Scoring**:
- Fuzzy matching: AI predictions vs ground truth
- Component scores: baselines, metrics, datasets
- Overall alignment percentage
- **Determines winner**: which approach matches humans better

### Running Options

**Basic Run**:
```bash
python experimental_setup_eval/run_pipeline.py papers_to_process.txt
```

**With Custom Output Directory**:
```bash
python experimental_setup_eval/run_pipeline.py \
    papers_to_process.txt \
    --output-dir runs/my_experiment
```

**Process Specific Papers**:
```bash
# Create custom list
echo "data/papers/collection/paper1/paper.md" > my_papers.txt
echo "data/papers/collection/paper2/paper.md" >> my_papers.txt

python experimental_setup_eval/run_pipeline.py my_papers.txt
```

---

## Viewing Results

### Output Structure

```
runs/{timestamp}_full_eval/
├── summary.json                    # Overall results & winners
└── {paper_id}/
    ├── paper_context.json          # Research context (AI sees)
    ├── ground_truth.json           # Extracted experiments
    ├── non_agentic_setup.json      # Non-agentic proposal
    ├── agentic_setup.json          # Agentic proposal
    ├── setup_comparison.json       # Quality comparison
    ├── non_agentic_alignment.json  # vs Ground Truth
    ├── agentic_alignment.json      # vs Ground Truth
    └── alignment_report.txt        # Human-readable report
```

### Quick Summary

```bash
cat runs/{timestamp}_full_eval/summary.json
```

Example:
```json
{
  "total_papers": 10,
  "non_agentic_wins": 4,
  "agentic_wins": 6,
  "avg_non_agentic_alignment": 0.65,
  "avg_agentic_alignment": 0.72,
  "avg_non_agentic_quality": 0.85,
  "avg_agentic_quality": 0.88,
  "total_cost": 3.20,
  "total_time": 850
}
```

### Detailed Report (Per Paper)

```bash
cat runs/{timestamp}_full_eval/{paper_id}/alignment_report.txt
```

Example:
```
Paper: Understanding Scaling Laws

QUALITY SCORES:
  Non-Agentic: 0.975
  Agentic:     0.967
  Winner:      Non-Agentic (tied, but faster)

ALIGNMENT WITH GROUND TRUTH:
  Non-Agentic: 85% (5/6 baselines, 4/4 metrics)
  Agentic:     78% (4/6 baselines, 3/4 metrics)
  Winner:      Non-Agentic

COST:
  Non-Agentic: 12s, 1 call
  Agentic:     58s, 3 calls (4.7x more expensive)

CONCLUSION: Non-agentic is better for this paper
```

### Quality Metrics Explained

**Setup Quality Score** (0-1):
- **Completeness** (30%): Has all required components?
- **Diversity** (20%): Baselines span complexity levels?
- **Alignment** (25%): Metrics match domain?
- **Datasets** (15%): Well-specified datasets?
- **Protocol** (10%): Rigorous experimental protocol?

**Alignment Score** (vs Ground Truth):
- **Baseline Recall**: % of true baselines found
- **Metric Recall**: % of true metrics found
- **Dataset Recall**: % of true datasets found
- **Overall**: Weighted average (baselines=40%, metrics=40%, datasets=20%)

---

## Cost Estimates

### Per Paper (with GPT-4o)

| Stage | Cost | Time |
|-------|------|------|
| Stage 1: Processing | ~$0.02 | 5s |
| Stage 2a: Non-Agentic | ~$0.05 | 12s |
| Stage 2b: Agentic (3 drafts) | ~$0.20 | 60s |
| Stage 3: Ground Truth | ~$0.03 | 8s |
| **Total** | **~$0.30** | **~85s** |

### Batch Processing

- **10 papers**: ~$3, ~15 minutes
- **50 papers**: ~$15, ~1 hour
- **100 papers**: ~$30, ~2 hours

### Cost Optimization

**Reduce agentic drafts**:
```python
# Edit run_pipeline.py around line 150
agentic_gen = AgenticSetupGenerator(num_drafts=2)  # Instead of 3
```

**Run non-agentic only** (for testing):
```python
# Comment out agentic generation section in run_pipeline.py
```

---

## Troubleshooting

### Common Issues

#### "No experiments found"

**Cause**: Paper lacks clear experimental sections

**Solutions**:
1. Use validation system to pre-filter papers
2. Check `ground_truth.json` to see what was extracted
3. Try papers with obvious "Experiments" or "Evaluation" sections

#### Low Alignment Scores

**Expected**: 20-60% alignment is normal

**Why**: AI cannot predict specific author choices without seeing experiments

**Note**: This is a research finding, not a bug. Different researchers make different valid choices.

#### "Invalid title detected" during validation

**Cause**: Title extraction picked up date or metadata

**Solutions**:
1. Check `validation_report.json` → `metadata.extracted_title`
2. Compare with `metadata.source_title` from metadata.json
3. Paper may have non-standard formatting

#### All Papers Failing Validation

**Cause**: Poor PDF-to-markdown conversion

**Solutions**:
1. Lower thresholds: `--min-abstract-length 50 --min-intro-length 200`
2. Check if papers are from ArXiv (should have better LaTeX conversion)
3. Try different papers from major venues (NeurIPS, ICML, etc.)

#### High Costs

**Solutions**:
1. Validate papers first to filter out poor quality
2. Reduce number of agentic drafts (see Cost Optimization)
3. Start with small batch (5 papers) to test
4. Use cheaper model for testing (edit model in generators)

#### API Rate Limits

**Cause**: Too many requests too fast

**Solutions**:
1. Add delays between papers (edit run_pipeline.py)
2. Use API keys with higher rate limits
3. Process in smaller batches

---

## Project Structure

```
AI-Scientist-v2-base/
├── experimental_setup_eval/        # Extension
│   ├── tools/
│   │   ├── paper_processor.py     # Parse & redact papers
│   │   └── ground_truth_extractor.py
│   ├── generators/
│   │   ├── non_agentic_generator.py
│   │   └── agentic_generator.py
│   ├── evaluation/
│   │   ├── quality_scorer.py
│   │   ├── comparator.py
│   │   └── alignment_scorer.py
│   ├── data_collection/
│   │   └── collect_papers.py      # Collection entry point
│   ├── run_pipeline.py            # Main pipeline
│   └── validate_and_filter_papers.py
│
├── data/papers/                   # Collected papers (immutable)
└── runs/                          # Pipeline results
```

---

## Research Questions

This pipeline helps answer:

1. **Quality**: Which approach generates better setups?
   - Metric: Setup quality scores (0-1)
   - Expected: Agentic slightly better due to multi-draft selection

2. **Alignment**: Which matches human choices better?
   - Metric: Alignment scores vs ground truth
   - Analysis: Does more exploration lead to better prediction?

3. **Cost-Benefit**: Is agentic worth the extra cost?
   - Analysis: Quality/alignment improvement vs 5-10x cost increase
   - Decision: When is extra cost justified?

4. **Paper Characteristics**: When does agentic help most?
   - Stratify by: domain, complexity, recency
   - Analysis: Complex papers may benefit more from multi-draft

---

## Next Steps

### 1. Test the System

```bash
# Quick test with 2 papers
python experimental_setup_eval/data_collection/collect_papers.py --query "attention mechanism" --max-results 2 --download-pdfs --convert-markdown
python experimental_setup_eval/run_pipeline.py data/papers/{timestamp}_collection/papers_to_process.txt
```

### 2. Scale Up

```bash
# Collect more papers
python experimental_setup_eval/data_collection/collect_papers.py --query "transformer" --max-results 20 --download-pdfs --convert-markdown

# Validate first
python experimental_setup_eval/validate_and_filter_papers.py data/papers/{timestamp}_collection/

# Process validated papers
python experimental_setup_eval/run_pipeline.py data/papers/{timestamp}_collection/validated_papers_to_process.txt
```

### 3. Analyze Results

- Compare non-agentic vs agentic win rates
- Calculate average alignment scores
- Perform cost-benefit analysis
- Stratify by paper domain or complexity

### 4. Publication

- Run on 50-100 papers for statistical significance
- Aggregate results across domains
- Generate figures and tables
- Write up findings

---

## Summary

**Complete workflow**:
```bash
# 1. Collect papers
python experimental_setup_eval/data_collection/collect_papers.py --query "your topic" --max-results 10 --download-pdfs --convert-markdown

# 2. Validate (recommended)
python experimental_setup_eval/validate_and_filter_papers.py data/papers/{timestamp}_collection/

# 3. Run pipeline
python experimental_setup_eval/run_pipeline.py data/papers/{timestamp}_collection/validated_papers_to_process.txt

# 4. View results
cat runs/{timestamp}_full_eval/summary.json
```

**What you get**:
- Setup quality scores
- Alignment scores (AI vs humans)
- Winner determination (non-agentic vs agentic)
- Cost and time analysis
- Per-paper detailed reports

**Research answer**: Can AI design experimental setups as well as humans?

---

## Support

For code reference:
- **Main Pipeline**: [run_pipeline.py](../experimental_setup_eval/run_pipeline.py)
- **Tools**: [tools/](../experimental_setup_eval/tools/)
- **Generators**: [generators/](../experimental_setup_eval/generators/)
- **Evaluation**: [evaluation/](../experimental_setup_eval/evaluation/)

For architecture details, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).
