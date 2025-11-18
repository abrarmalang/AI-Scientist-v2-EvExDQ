# Experimental Setup Evaluation

**Evaluate how well AI can design experimental setups for research papers.**

This extension to [AI Scientist v2](https://github.com/SakanaAI/AI-Scientist-v2) compares AI-generated experimental setups against human choices to answer: **Can AI design experiments as well as humans?**

---

## Quick Start

### 1. Install Dependencies

```bash
cd AI-Scientist-v2-EvExDQ
pip install -r requirements.txt

# Set at least one API key (pipeline will auto-select model)
export OPENAI_API_KEY="your-key"        # For GPT-4o
# OR
export ANTHROPIC_API_KEY="your-key"     # For Claude Sonnet
```

**Note**: You only need one API key. The pipeline automatically uses whichever is available.

### 2. Collect Papers

```bash
python experimental_setup_eval/data_collection/collect_papers.py \
    --query "transformer neural networks" \
    --max-results 5 \
    --download-pdfs \
    --convert-markdown
```

**Output**: `data/papers/{timestamp}_collection/` containing papers and `papers_to_process.txt`

### 3. Validate Papers (Recommended)

Before running the expensive evaluation pipeline, validate that papers were extracted correctly:

```bash
python experimental_setup_eval/validate_and_filter_papers.py \
    data/papers/{timestamp}_collection
```

**Output**:
- `validated_papers_to_process.txt` - Only papers with successful extraction
- `validation_report.json` - Detailed validation results

**Why validate?** Catches papers with poor PDF-to-markdown conversion, missing sections, or incomplete extraction. Saves money by preventing wasted LLM calls on bad papers.

### 4. Run Evaluation Pipeline

```bash
# Use validated papers (recommended)
python experimental_setup_eval/run_pipeline.py \
    data/papers/{timestamp}_collection/validated_papers_to_process.txt

# OR use all papers (not recommended)
python experimental_setup_eval/run_pipeline.py \
    data/papers/{timestamp}_collection/papers_to_process.txt
```

**Output**: `runs/{timestamp}_full_eval/` with complete results

### 5. View Results

```bash
# Quick summary
cat runs/{timestamp}_full_eval/summary.json

# Detailed report for a paper
cat runs/{timestamp}_full_eval/{paper_id}/alignment_report.txt
```

---

## What It Does

The pipeline evaluates experimental design quality through **3 main stages** (plus optional validation):

### Stage 0: Validation (Optional but Recommended)
- Validates extraction quality using actual PaperProcessorTool
- Checks for complete sections (title, abstract, introduction, method)
- Filters out papers with poor PDF-to-markdown conversion
- Saves costs by preventing wasted LLM calls
- Outputs: `validated_papers_to_process.txt`, `validation_report.json`

### Stage 1: Paper Processing
- Uses enhanced **PaperProcessorTool** with HTML cleaning and improved parsing
- Extracts research context (title, abstract, **introduction**, **related work**, method)
- **Redacts** experimental sections (AI never sees them)
- Identifies research question and domain
- **Key Feature**: Passes introduction and related work (containing specific baseline names) to generators
- Outputs: `paper_context.json`

### Stage 2: Setup Generation (Dual-Mode)
**Non-Agentic**:
- Single LLM call (fast, cheap ~$0.05, 12s)
- Proposes: baselines, metrics, datasets, experimental protocol

**Agentic**:
- Multi-draft approach (3 drafts with different focus)
- Quality scoring and selection (slower ~$0.20, 60s)
- More thorough exploration

Both approaches output structured experimental setups.

### Stage 3: Ground Truth Comparison
- Extracts actual experiments from paper
- Compares AI predictions vs human choices (fuzzy matching)
- Scores alignment:
  - Baseline recall: % of true baselines found
  - Metric recall: % of true metrics found
  - Dataset recall: % of true datasets found
- **Determines winner**: Which approach matches humans better?
- Outputs: `alignment_report.txt`, `summary.json`

---

## Results You Get

### Per Paper

```
Paper: Understanding Scaling Laws

QUALITY SCORES:
  Non-Agentic: 0.975
  Agentic:     0.967
  Winner:      Non-Agentic (quality tied, but faster)

ALIGNMENT WITH GROUND TRUTH:
  Non-Agentic: 85% (5/6 baselines, 4/4 metrics)
  Agentic:     78% (4/6 baselines, 3/4 metrics)
  Winner:      Non-Agentic

COST:
  Non-Agentic: 12s, 1 call
  Agentic:     58s, 3 calls (4.7x more expensive)

CONCLUSION: Non-agentic is better for this paper
```

### Overall Summary

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

---

## Project Structure

```
experimental_setup_eval/
├── README.md                       # This file
├── validate_and_filter_papers.py  # Validation ⭐
├── run_pipeline.py                 # Main pipeline ⭐
│
├── docs/                           # Detailed documentation
│   ├── USER_GUIDE.md              # Complete user guide
│   ├── DEVELOPER_GUIDE.md         # Architecture & development
│   ├── ARCHITECTURE.md            # AI Scientist v2 reference
│   └── ADVANCED_GUIDES.md         # Advanced topics
│
├── tools/                          # Core processing tools
│   ├── paper_processor.py         # Parse & redact papers
│   └── ground_truth_extractor.py  # Extract actual experiments
│
├── generators/                     # Setup generation
│   ├── non_agentic_generator.py   # Single-shot generation
│   └── agentic_generator.py       # Multi-draft generation
│
├── evaluation/                     # Scoring & comparison
│   ├── quality_scorer.py          # Score setup quality
│   ├── comparator.py              # Compare approaches
│   └── alignment_scorer.py        # Alignment vs ground truth
│
├── data_collection/                # Collection internals
│   ├── collect_papers.py          # Paper collection ⭐
│   └── improved_paper_extraction.py
│
└── notebooks/                      # Jupyter notebooks
    └── visualize_results.ipynb    # Results visualization
```

**Main Entry Points** (⭐):
- `data_collection/collect_papers.py` - Collect papers with improved text extraction
- `validate_and_filter_papers.py` - Validate paper extraction quality
- `run_pipeline.py` - Run full evaluation pipeline

---

## Output Structure

### Validation Output (Stage 0)
```
data/papers/{timestamp}_collection/
├── validated_papers_to_process.txt  # Filtered list of quality papers
└── validation_report.json           # Detailed validation results
```

### Pipeline Output (Stages 1-3)
```
runs/{timestamp}_full_eval/
├── summary.json                    # Overall results & winners
└── {paper_id}/
    ├── paper_context.json          # Research context (AI input)
    ├── ground_truth.json           # Actual experiments
    ├── non_agentic_setup.json      # Non-agentic proposal
    ├── agentic_setup.json          # Agentic proposal
    ├── setup_comparison.json       # Quality comparison
    ├── non_agentic_alignment.json  # Alignment vs ground truth
    ├── agentic_alignment.json      # Alignment vs ground truth
    └── alignment_report.txt        # Human-readable report
```

---

## Validation System

### Why Validation Matters

**Problem**: PDF-to-markdown conversion can fail, producing papers with:
- Missing sections (no abstract, no introduction)
- Malformed titles (extracted as "Unknown Title")
- Poor section parsing (all content in one giant section)

**Solution**: The validator uses the **actual PaperProcessorTool** to check if extraction succeeded before running expensive LLM calls.

### Validation Checks

Each paper is scored on:
- ✓ Markdown exists and is readable
- ✓ Title extracted correctly (not "Unknown Title")
- ✓ Abstract ≥ 100 characters
- ✓ Introduction ≥ 300 characters
- ✓ Method section ≥ 200 characters
- ✓ Total content ≥ 1000 characters
- ⚠ Related work present (warning only)

### Custom Thresholds

```bash
python experimental_setup_eval/validate_and_filter_papers.py \
    data/papers/{timestamp}_collection \
    --min-abstract-length 200 \
    --min-intro-length 500 \
    --min-method-length 300
```

### Example Validation Results

From a collection of 3 weather forecasting papers:
```
✓ PASS (90%): FengWu-GHR - All sections extracted
✗ FAIL (50%): ClimODE - Poor PDF conversion, no sections
✗ FAIL (70%): Pangu-Weather - Missing abstract/introduction

Validated: 1/3 papers passed
```

**Outcome**: Only FengWu-GHR processed, saving 2 expensive pipeline runs on broken papers.

---

## Complete Workflow Example

```bash
# Step 1: Collect papers
python experimental_setup_eval/data_collection/collect_papers.py \
    --query "weather forecasting" \
    --max-results 5 \
    --download-pdfs \
    --convert-markdown

# Output: data/papers/20251119_161731_collection/

# Step 2: Validate extraction quality (recommended!)
python experimental_setup_eval/validate_and_filter_papers.py \
    data/papers/20251119_161731_collection

# Output:
# ✓ PASS (90%): FengWu-GHR
# ✗ FAIL (50%): ClimODE - Poor PDF conversion
# Validated: 1/3 papers passed

# Step 3: Run pipeline on validated papers only
python experimental_setup_eval/run_pipeline.py \
    data/papers/20251119_161731_collection/validated_papers_to_process.txt

# Output: runs/20251119_162419_full_eval/

# Step 4: View results
cat runs/20251119_162419_full_eval/summary.json
cat runs/20251119_162419_full_eval/FengWu-GHR_*/alignment_report.txt
```

**Result**: Only high-quality papers are processed, saving money and improving results!

---

## Cost Estimates

**Per Paper** (with GPT-4o):
- Stage 1 (Processing): ~$0.02, 5s
- Stage 2a (Non-Agentic): ~$0.05, 12s
- Stage 2b (Agentic): ~$0.20, 60s
- Stage 3 (Ground Truth): ~$0.03, 8s
- **Total**: ~$0.30, ~85s per paper

**Batch Processing**:
- 10 papers: ~$3, ~15 minutes
- 50 papers: ~$15, ~1 hour
- 100 papers: ~$30, ~2 hours

---

## Research Question

**Can AI design experimental setups as well as humans?**

We answer this by:
1. Generating setups using two AI approaches (non-agentic vs agentic)
2. Comparing against actual author choices (ground truth)
3. Measuring alignment and quality
4. Analyzing cost-benefit trade-offs

**Key Insight**: Determines which AI approach produces setups closer to human experimental design.

---

## Documentation

### Quick Reference

- **Getting Started** → This README
- **Complete User Guide** → [docs/USER_GUIDE.md](docs/USER_GUIDE.md)
  - Detailed instructions for collection, validation, pipeline, results
  - Troubleshooting common issues
- **Developer Guide** → [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)
  - Architecture overview
  - How to extend the system
  - Adding new features
- **AI Scientist v2 Reference** → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
  - Core AI Scientist v2 architecture
- **Advanced Topics** → [docs/ADVANCED_GUIDES.md](docs/ADVANCED_GUIDES.md)
  - Manual reproduction with ChatGPT/Claude
  - Prompt improvement details
  - Marker PDF integration

### Documentation Structure

All detailed documentation is in the `docs/` folder:

```
docs/
├── USER_GUIDE.md          # Everything for users/operators
├── DEVELOPER_GUIDE.md     # Everything for developers
├── ARCHITECTURE.md        # AI Scientist v2 core reference
└── ADVANCED_GUIDES.md     # Advanced/niche topics
```

---

## Troubleshooting

### Common Issues

**"No experiments found"**
- Paper may lack clear experimental sections
- Try validation first to filter out poor quality papers
- Check `ground_truth.json` to see what was extracted

**Low alignment scores (20-60%)**
- This is normal! AI cannot predict specific author choices without seeing experiments
- This is a research finding, not a bug
- Different researchers make different valid choices

**All papers failing validation**
- PDF-to-markdown conversion produced poor results
- Lower thresholds: `--min-abstract-length 50 --min-intro-length 200`
- Try papers from major venues (NeurIPS, ICML, etc.)

**High costs**
- Use validation to filter out poor papers first
- Reduce agentic drafts: Edit `run_pipeline.py`, change `num_drafts=3` to `num_drafts=2`
- Start with small batch (5 papers) to test

---

## Advanced Usage

### Custom Parameters

Edit `run_pipeline.py` to adjust:
- `num_drafts=3` → Change agentic draft count
- Model selection (default: GPT-4o, Claude Sonnet)
- Redaction keywords

### Visualization

Use Jupyter notebooks for rich visualizations:
```bash
jupyter notebook experimental_setup_eval/notebooks/visualize_results.ipynb
```

---

## Support

**Main Pipeline**: `experimental_setup_eval/run_pipeline.py`
**Validation**: `experimental_setup_eval/validate_and_filter_papers.py`
**AI Scientist v2**: https://github.com/SakanaAI/AI-Scientist-v2

For detailed documentation, see the [docs/](docs/) folder.
