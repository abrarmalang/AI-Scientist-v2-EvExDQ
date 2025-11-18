# AI Scientist-v2 Architecture Documentation

## Overview

The **AI Scientist-v2** is an autonomous scientific research system that generates research ideas, conducts machine learning experiments through intelligent tree search, analyzes results, and writes complete research papers. It's designed for **workshop-level automated scientific discovery** and notably generated the first AI-written paper accepted through peer review.

### What It Does

1. **Generates research ideas** from high-level topics using LLMs with tool access
2. **Conducts experiments** through Best-First Tree Search (BFTS)
3. **Analyzes results** and generates visualizations
4. **Writes complete research papers** in LaTeX with citations
5. **Reviews generated papers** using LLM and Vision-Language Models

### Key Difference from v1

Unlike its predecessor (AI Scientist-v1), v2:
- Removes reliance on human-authored templates
- Generalizes across ML domains
- Employs progressive agentic tree search
- Takes a broader, more exploratory approach (lower success rates but more creative)

### Domain Applicability

**Current Focus: Machine Learning Research**

The AI Scientist-v2 is currently **optimized for ML/AI research**, with:
- Stage names aligned to ML methodology (baseline tuning, ablation studies)
- Prompts referencing datasets, baselines, hyperparameters, training dynamics
- Examples and templates focused on ML experiments
- Metrics designed for ML evaluation (accuracy, loss, etc.)

**Core Architecture: Domain-Agnostic**

However, the **fundamental architecture is general-purpose**:
- **Tree Search (BFTS)**: Explores solution spaces by generating code → executing → evaluating → improving
- **Journal/Node**: Generic structure for tracking code, metrics, plots, and analysis
- **Agent/Interpreter**: Executes arbitrary Python code with no ML-specific requirements
- **Metric System**: Can track any numerical or structured metrics

**Adaptability to Other Computational Fields**

The system can be adapted to other research domains **primarily through prompt engineering**:

| Field | Adaptations Required |
|-------|---------------------|
| **Theoretical CS/Math** | Replace "datasets" → "problem instances"; "baselines" → "existing algorithms"; metrics → runtime complexity, proof length |
| **Systems Research** | Replace "datasets" → "workloads/benchmarks"; metrics → throughput, latency, resource utilization |
| **Physics/Chemistry** | Replace "datasets" → "simulation parameters"; experiments → computational models; metrics → physical quantities |
| **Biology/Bioinformatics** | Replace "datasets" → "genomic datasets"; "baselines" → "existing analysis pipelines"; metrics → biological significance |

**Key Requirement**: The research must be **computational** - experiments must be implementable and evaluatable through code execution.

**What Changes for Other Domains**:
1. **Prompts**: Update stage descriptions and experimental vocabulary (minimal code changes)
2. **Evaluation Criteria**: Domain-specific standards for "good" experiments
3. **Tools**: Add domain-specific tools (e.g., theorem provers for math, simulators for physics)
4. **Examples**: Provide domain-specific few-shot examples

**What Stays the Same**:
- Tree search algorithm (BFTS)
- Code generation → execution → evaluation loop
- Multi-stage progression framework
- Paper writing and review infrastructure

**Bottom Line**: The system is **architecturally domain-agnostic** but **currently ML-optimized**. Extending to other computational research fields requires primarily prompt engineering rather than core system modifications.

---

## Core Architecture Overview

### Two-Phase Pipeline

#### **Phase 1: Ideation**
**Entry Point**: [`ai_scientist/perform_ideation_temp_free.py`](../ai_scientist/perform_ideation_temp_free.py)

- **Input**: Research topic description (markdown file)
- **Process**: LLM with tool access (Semantic Scholar for literature search)
- **Output**: JSON file with structured research proposals

**Flow**:
```
my_research_topic.md
  ↓
  LLM generates ideas
  ↓
  Semantic Scholar novelty checking
  ↓
  Iterative refinement (num_reflections rounds)
  ↓
my_research_topic.json
```

#### **Phase 2: Execution**
**Entry Point**: [`launch_scientist_bfts.py`](../launch_scientist_bfts.py)

- **Input**: Research ideas JSON + configuration
- **Process**: Multi-stage experimental workflow with tree search
- **Output**: Research paper PDF + experimental artifacts

**Flow**:
```
my_research_topic.json + bfts_config.yaml
  ↓
  Stage 1: Initial Implementation
  ↓
  Stage 2: Baseline Tuning
  ↓
  Stage 3: Creative Research
  ↓
  Stage 4: Ablation Studies
  ↓
  Plot Aggregation
  ↓
  Paper Writing (with citations)
  ↓
  Automated Review
  ↓
timestamp_ideaname.pdf
```

---

## Key Abstractions & Classes

### 1. AgentManager

**Location**: [`ai_scientist/treesearch/agent_manager.py`](../ai_scientist/treesearch/agent_manager.py)

**Purpose**: Top-level orchestrator for the entire experimental workflow.

#### Key Classes

##### `Stage`
Represents a phase of research.

**Attributes**:
- `name`: Stage identifier
- `description`: What this stage accomplishes
- `goals`: List of completion criteria
- `max_iterations`: Maximum iterations for this stage
- `num_drafts`: Number of initial solution attempts
- `stage_number`: 1=initial_implementation, 2=baseline_tuning, 3=creative_research, 4=ablation_studies

##### `StageTransition`
Records transitions between stages with reasoning.

**Attributes**:
- `from_stage`: Previous stage name
- `to_stage`: Next stage name
- `reason`: LLM-generated explanation for transition

##### `AgentManager`
Main orchestration class that manages the complete experimental workflow.

**Key Attributes**:
- `task_desc`: Research idea description (Title, Abstract, Hypothesis, Experiments)
- `stages`: List of Stage objects
- `journals`: Dict mapping stage names to Journal objects
- `stage_history`: List of StageTransition objects
- `main_stage_dict`: Maps stage numbers to names

**Key Methods**:
- `run()`: Main execution loop through stages and sub-stages
- `_create_agent_for_stage(stage, substage_goal)`: Creates ParallelAgent for each stage
- `_check_stage_completion(stage, journal)`: Determines if stage goals are met using LLM
- `_check_substage_completion(substage_goal, journal)`: Evaluates sub-stage completion
- `_generate_substage_goal(stage, journal)`: Uses LLM to generate next sub-stage goals
- `_get_best_implementation(stage_name)`: Retrieves best node from completed stage

#### The 4-Stage Research Process

1. **Stage 1: Initial Implementation**
   - Goal: Get basic working code on simple datasets
   - Focus: Correctness over performance

2. **Stage 2: Baseline Tuning**
   - Goal: Optimize hyperparameters with multiple datasets
   - Focus: Establish strong baselines

3. **Stage 3: Creative Research**
   - Goal: Explore novel improvements and experiments
   - Focus: Innovation and discovery

4. **Stage 4: Ablation Studies**
   - Goal: Systematic component analysis
   - Focus: Understanding what works and why

---

### 2. ParallelAgent

**Location**: [`ai_scientist/treesearch/parallel_agent.py`](../ai_scientist/treesearch/parallel_agent.py)

**Purpose**: Executes parallel exploration of the solution space using Best-First Tree Search (BFTS).

#### Key Classes

##### `MinimalAgent`
Base agent with prompt generation capabilities.

**Responsibilities**:
- Provides environment setup instructions
- Implementation guidelines
- Evaluation requirements
- Error handling patterns

##### `ParallelAgent`
Main agent for parallel experiment execution.

**Key Features**:
- Manages **multiple worker threads** (`num_workers` in config)
- Implements **Best-First Tree Search (BFTS)** algorithm
- Handles code generation, execution, debugging, and improvement cycles

**Node Operations**:
1. **Draft**: Generate initial solutions (`num_drafts` initial nodes)
2. **Improve**: Refine successful nodes based on feedback
3. **Debug**: Fix buggy implementations

**Key Attributes**:
- `journal`: Stores solution tree
- `num_workers`: Number of parallel exploration paths
- `steps`: Maximum iterations
- `gpu_manager`: Allocates GPUs across workers

**Important Methods**:
- `step()`: Execute one iteration of tree search
  - Selects best leaf nodes to expand
  - Generates improvements or debugs failures
  - Executes code and collects metrics

- `_run_multi_seed_evaluation(node)`: Evaluate best node with multiple random seeds

- `_run_plot_aggregation(node)`: Aggregate plots from seed runs

- `_generate_draft()`: Create initial solution drafts
  - Generates `num_drafts` independent implementations
  - Each uses LLM to create code from scratch

- `_improve_node(node)`: Generate improvements for good nodes
  - Analyzes current implementation
  - Proposes specific enhancements
  - Creates child node with improved code

- `_debug_node(node)`: Fix buggy implementations
  - Analyzes error messages and stack traces
  - Generates fix using LLM
  - Creates child node with corrected code
  - Respects `max_debug_depth` limit

##### `GPUManager`
Manages GPU allocation across parallel workers.

##### `AblationConfig`
Tracks ablation experiment state.

##### `HyperparamTuningIdea`
Represents hyperparameter tuning experiments.

#### Tree Search Algorithm

```
Initialize with num_drafts initial nodes
  ↓
Loop for 'steps' iterations:
  ├─ Select best leaf nodes (up to num_workers)
  ├─ For each selected node in parallel:
  │   ├─ If buggy and debug_prob > random():
  │   │   └─ Debug node (if debug_depth < max_debug_depth)
  │   └─ Else:
  │       └─ Improve node (generate enhancement)
  │
  ├─ Execute all new nodes via Interpreter
  ├─ Extract metrics via metric_parse_spec
  ├─ Analyze plots via vlm_feedback_spec
  └─ Add nodes to Journal
  ↓
Select best node via Journal.get_best_node()
```

---

### 3. Journal & Node

**Location**: [`ai_scientist/treesearch/journal.py`](../ai_scientist/treesearch/journal.py)

**Purpose**: The solution tree data structure that maintains complete experimental history.

#### `Node` Class

Represents a single experiment/implementation in the solution tree.

**Core Attributes**:
- `code`: Python implementation (string)
- `plan`: High-level solution description
- `parent`: Parent Node (or None for root)
- `children`: List of child Nodes
- `metric`: Performance metric (MetricValue object)
- `is_buggy`: Whether execution failed (boolean)
- `analysis`: Post-execution analysis from LLM
- `exec_time`: Execution duration in seconds
- `plot_paths`: List of paths to generated visualizations
- `vlm_feedback_summary`: Vision-Language Model feedback on plots
- `datasets_successfully_tested`: List of datasets tested
- `stdout`: Captured standard output
- `stderr`: Captured standard error

**Key Properties**:
- `stage_name`: Returns "draft", "debug", or "improve" based on node type
- `debug_depth`: Number of consecutive debugging steps from root
- `is_leaf`: Whether node has children (boolean)
- `is_root`: Whether node is root (boolean)

**Methods**:
- `add_child(child_node)`: Add child to tree
- `get_ancestors()`: Return list of ancestor nodes
- `to_dict()`: Serialize to dictionary

#### `Journal` Class

Collection of nodes representing the complete solution tree.

**Key Attributes**:
- `nodes`: List of all Node objects
- `root_nodes`: Nodes without parents (initial drafts)

**Key Methods**:
- `get_best_node()`: **Most Important Method**
  - Uses LLM to select best implementation
  - Considers:
    - Metrics across datasets
    - Training dynamics (from plots)
    - Plot quality (VLM feedback)
    - Code quality and complexity
  - Returns Node object

- `generate_summary()`: LLM-based summary of experimental progress
  - Used for stage completion checks
  - Provides overview of what's been tried

- `add_node(node)`: Add node to journal

- `good_nodes`: Property returning list of non-buggy nodes

- `buggy_nodes`: Property returning list of buggy nodes

- `draft_nodes`: Property returning initial draft nodes

**Tree Structure Example**:
```
Draft 1 (root)
  ├─ Improve 1.1
  │   ├─ Improve 1.1.1
  │   └─ Debug 1.1.2 (buggy)
  └─ Debug 1.2
      └─ Improve 1.2.1

Draft 2 (root)
  └─ Improve 2.1

Draft 3 (root, buggy)
```

---

### 4. Interpreter

**Location**: [`ai_scientist/treesearch/interpreter.py`](../ai_scientist/treesearch/interpreter.py)

**Purpose**: Safe Python code execution sandbox for LLM-generated code.

#### `ExecutionResult` Class

Captures execution output.

**Attributes**:
- `stdout`: Standard output (string)
- `stderr`: Standard error (string)
- `exception`: Exception object if raised
- `exec_time`: Execution time in seconds

#### `Interpreter` Class

Manages code execution in isolated processes.

**Key Features**:
- **Timeout enforcement**: Configurable (default 1 hour via `bfts_config.yaml`)
- **Stdout/stderr capture**: Complete output logging
- **Exception handling**: Full stack traces preserved
- **Working directory isolation**: Each execution in separate directory
- **Process isolation**: Runs in subprocess for safety

**Key Methods**:
- `run(code, timeout=None)`: Execute Python code
  - Returns ExecutionResult
  - Kills process if timeout exceeded
  - Captures all output streams

**Why It Matters**: LLM-generated code can contain:
- Infinite loops
- Dangerous system calls
- Resource exhaustion
- Unintended side effects

The Interpreter provides critical safety guarantees.

**Usage Example**:
```python
interpreter = Interpreter(work_dir="/path/to/workdir")
result = interpreter.run(code_string, timeout=3600)

if result.exception:
    print(f"Execution failed: {result.exception}")
else:
    print(f"Success! Output: {result.stdout}")
```

---

### 5. Backend System

**Location**: [`ai_scientist/treesearch/backend/`](../ai_scientist/treesearch/backend/)

**Purpose**: Abstraction layer for multiple LLM providers with unified interface.

#### Key Components

**`backend_openai.py`**: OpenAI API integration
- Supports GPT-4o, o1, o3-mini models
- Function calling support
- Streaming and batch APIs

**`backend_anthropic.py`**: Anthropic/Claude API integration
- Supports Claude 3.5 Sonnet
- Tool use (Anthropic's function calling)
- Supports multiple access methods (API, Bedrock, Vertex AI)

**`utils.py`**: Core utilities
- `FunctionSpec`: Defines function calling schemas
- `compile_prompt_to_md()`: Converts prompts to markdown
- `query()`: **Unified interface for LLM calls**

#### `FunctionSpec` Class

Defines structured output schemas for reliable JSON parsing from LLMs.

**Attributes**:
- `name`: Function name
- `description`: What the function does
- `parameters`: JSON schema for parameters

**Built-in Function Specs**:

1. **`review_func_spec`**: Bug detection and code analysis
   ```json
   {
     "has_bug": boolean,
     "bug_description": string,
     "suggestions": [string]
   }
   ```

2. **`vlm_feedback_spec`**: Plot quality analysis
   ```json
   {
     "plot_quality": 1-10,
     "insights": string,
     "suggestions": [string]
   }
   ```

3. **`metric_parse_spec`**: Extract metrics from execution logs
   ```json
   {
     "metric_name": string,
     "metric_value": float,
     "dataset": string
   }
   ```

4. **`plot_selection_spec`**: Choose best plots for paper
   ```json
   {
     "selected_plots": [string],
     "rationale": string
   }
   ```

**Usage**:
```python
from ai_scientist.treesearch.backend.utils import query, review_func_spec

response = query(
    messages=[{"role": "user", "content": "Analyze this code..."}],
    model="gpt-4o",
    func_spec=review_func_spec,
    temp=0.5
)
# Returns structured JSON matching review_func_spec schema
```

---

### 6. Configuration System

**Location**: [`ai_scientist/treesearch/utils/config.py`](../ai_scientist/treesearch/utils/config.py)

**Purpose**: Centralized configuration management via [`bfts_config.yaml`](../bfts_config.yaml).

#### Key Classes

##### `Config`
Main configuration dataclass.

**Attributes**:
- `data_dir`: Data location
- `log_dir`: Logging output directory
- `workspace_dir`: Working directory for experiments
- `exec`: Execution settings (ExecConfig)
- `agent`: Agent configuration (AgentConfig)
- `experiment`: Experiment settings (ExperimentConfig)

##### `ExecConfig`
Execution-specific settings.

**Attributes**:
- `timeout`: Code execution timeout (seconds)
- `main_file_name`: Name of main experiment file (default: "experiment.py")
- `plot_file_name`: Name of plotting script (default: "plot.py")

##### `AgentConfig`
Agent behavior configuration.

**Attributes**:
- `steps`: Maximum iterations per stage
- `num_workers`: Number of parallel exploration threads
- `num_seeds`: Number of random seeds for final evaluation
- `k_fold_validation`: Cross-validation folds (not currently used)
- `code`: ModelConfig for code generation
- `feedback`: ModelConfig for evaluation/feedback
- `vlm_feedback`: ModelConfig for plot analysis
- `search`: SearchConfig for tree search parameters

##### `ModelConfig`
LLM model settings.

**Attributes**:
- `model`: Model identifier (e.g., "claude-3-5-sonnet-20241022")
- `temp`: Temperature (0.0 to 2.0)
- `max_tokens`: Maximum output tokens
- `n`: Number of samples (for ensemble)

##### `SearchConfig`
Tree search parameters.

**Attributes**:
- `max_debug_depth`: Maximum consecutive debugging attempts per branch
- `debug_prob`: Probability of debugging vs. new exploration (0.0 to 1.0)
- `num_drafts`: Number of initial draft nodes

##### `ExperimentConfig`
Experiment-specific settings.

**Attributes**:
- `num_syn_datasets`: Required number of datasets to test
- `dataset_names`: Optional list of specific datasets

#### Example Configuration (`bfts_config.yaml`)

```yaml
agent:
  steps: 5                    # Max iterations per stage
  num_workers: 4              # Parallel exploration paths
  num_seeds: 3                # Seeds for final evaluation
  k_fold_validation: 1        # Cross-validation folds

  code:
    model: claude-3-5-sonnet-20241022
    temp: 1.0
    max_tokens: 12000

  feedback:
    model: gpt-4o-2024-11-20
    temp: 0.5
    max_tokens: 2000

  vlm_feedback:
    model: gpt-4o-2024-11-20
    temp: 0.5

  search:
    max_debug_depth: 3        # Max debugging attempts
    debug_prob: 0.5           # 50% chance of debugging buggy nodes
    num_drafts: 3             # Initial solution attempts

exec:
  timeout: 3600               # 1 hour execution timeout
  main_file_name: experiment.py
  plot_file_name: plot.py

experiment:
  num_syn_datasets: 1         # Required number of datasets
```

---

### 7. LLM Integration

**Location**: [`ai_scientist/llm.py`](../ai_scientist/llm.py)

**Purpose**: Unified LLM calling interface across multiple providers.

#### Key Functions

##### `create_client(model)`
Creates appropriate client based on model name.

**Returns**: Client object (OpenAI, Anthropic, Bedrock, etc.)

**Supported Providers**:
- **OpenAI**: `gpt-4o-*`, `o1-*`, `o3-mini-*`
- **Anthropic**: `claude-*` (via API, Bedrock, Vertex AI)
- **Google**: `gemini-*`
- **Ollama**: `deepseek-*`, `llama-*`, `qwen-*`

##### `get_response_from_llm(messages, client, model, ...)`
Single LLM call with message history.

**Parameters**:
- `messages`: List of {"role": "user/assistant", "content": "..."}
- `client`: Client from `create_client()`
- `model`: Model identifier
- `system_message`: Optional system prompt
- `print_debug`: Print request/response for debugging
- `msg_history`: Include previous messages
- `temperature`: Sampling temperature
- `max_tokens`: Maximum output tokens

**Returns**: String response from LLM

##### `get_batch_responses_from_llm(..., n=3)`
Ensemble responses (multiple samples).

**Parameters**: Same as `get_response_from_llm()` plus:
- `n`: Number of samples to generate

**Returns**: List of n string responses

##### `extract_json_between_markers(text)`
Parse JSON from LLM outputs.

**Handles**:
- Markdown code blocks: ` ```json ... ``` `
- JSON markers: `[JSON]...[/JSON]`
- Raw JSON in text

**Returns**: Parsed dictionary or None

#### Usage Example

```python
from ai_scientist.llm import create_client, get_response_from_llm

client = create_client("gpt-4o-2024-11-20")

messages = [
    {"role": "user", "content": "Explain gradient descent"}
]

response = get_response_from_llm(
    messages=messages,
    client=client,
    model="gpt-4o-2024-11-20",
    system_message="You are a helpful ML expert.",
    temperature=0.7,
    max_tokens=1000
)

print(response)
```

---

### 8. Paper Writing System

**Purpose**: Generate complete research papers from experimental results.

#### Main Components

##### `perform_writeup.py` / `perform_icbinb_writeup.py`

**Location**: [`ai_scientist/perform_writeup.py`](../ai_scientist/perform_writeup.py)

**Purpose**: Generate LaTeX papers (8-page full or 4-page workshop format).

**Key Functions**:

1. **`perform_writeup(idea, folder_name, ...)`**
   - Generates complete 8-page research paper
   - Iteratively writes sections (Introduction, Methods, Results, etc.)
   - Integrates experimental results and plots
   - Returns path to generated LaTeX

2. **`perform_icbinb_writeup(idea, folder_name, ...)`**
   - Generates 4-page workshop paper
   - Condensed format for "I Can't Believe It's Not Better" workshop

3. **`gather_citations(client, model, draft, num_cite_rounds)`**
   - **Iterative citation collection** using Semantic Scholar
   - Process:
     1. LLM identifies claims needing citations
     2. Semantic Scholar search for relevant papers
     3. LLM selects best citations
     4. Insert citations into draft
     5. Repeat for `num_cite_rounds`
   - Returns updated LaTeX with bibliography

4. **`compile_latex(cwd, pdf_file, timeout)`**
   - Compiles LaTeX to PDF
   - Runs `pdflatex` multiple times
   - Runs `bibtex` for bibliography
   - Error handling and logging
   - Returns success/failure status

**Writing Process**:
```
Experimental results + plots
  ↓
Generate Abstract
  ↓
Generate Introduction
  ↓
Generate Methods
  ↓
Generate Results
  ↓
Generate Discussion
  ↓
Generate Conclusion
  ↓
Gather citations (iterative)
  ↓
Compile LaTeX → PDF
```

##### `perform_plotting.py`

**Location**: [`ai_scientist/perform_plotting.py`](../ai_scientist/perform_plotting.py)

**Key Function**: `aggregate_plots(folder_name, model)`
- Collects plots from all experimental stages
- Uses LLM to select most relevant visualizations
- Considers plot diversity and informativeness
- Returns list of selected plot paths

##### `perform_llm_review.py` / `perform_vlm_review.py`

**Location**: [`ai_scientist/perform_llm_review.py`](../ai_scientist/perform_llm_review.py)

**Purpose**: Automated paper review.

**Key Functions**:

1. **`perform_review(text, model, ...)`**
   - Text-based review of paper content
   - Evaluates:
     - Novelty and significance
     - Technical correctness
     - Clarity and organization
     - Experimental rigor
   - Returns review as JSON

2. **`perform_imgs_cap_ref_review(folder_name, pdf_file, model)`**
   - Vision-Language Model review
   - Analyzes:
     - Figure quality and relevance
     - Caption appropriateness
     - Reference formatting
   - Returns review as JSON

---

### 9. Tools System

**Location**: [`ai_scientist/tools/`](../ai_scientist/tools/)

**Purpose**: Extensible tool framework for LLM agents during ideation.

#### Base Architecture

##### `BaseTool` (Abstract Base Class)

**Location**: [`ai_scientist/tools/base_tool.py`](../ai_scientist/tools/base_tool.py)

**Attributes**:
- `name`: Tool identifier
- `description`: What the tool does (for LLM)
- `parameters`: Parameter schema

**Abstract Method**:
- `use_tool(**kwargs)`: Execute tool functionality

#### Concrete Implementations

##### `SemanticScholarSearchTool`

**Location**: [`ai_scientist/tools/semantic_scholar_search.py`](../ai_scientist/tools/semantic_scholar_search.py)

**Purpose**: Literature search via Semantic Scholar API.

**Key Features**:
- Searches academic papers by query string
- Returns ranked results by citation count
- Backoff retry logic for API rate limiting
- Respects S2_API_KEY environment variable

**Parameters**:
- `query`: Search query string
- `limit`: Maximum number of results (default: 10)

**Returns**: List of paper dictionaries with:
- `title`: Paper title
- `authors`: Author list
- `year`: Publication year
- `citationCount`: Number of citations
- `abstract`: Paper abstract
- `url`: Semantic Scholar URL

**Usage During Ideation**:
```python
tool = SemanticScholarSearchTool()

results = tool.use_tool(
    query="transformer attention mechanisms",
    limit=5
)

for paper in results:
    print(f"{paper['title']} ({paper['citationCount']} citations)")
```

#### Tool Integration in Ideation

During the ideation phase ([`perform_ideation_temp_free.py`](../ai_scientist/perform_ideation_temp_free.py)):
1. LLM generates research idea
2. Tool access provided for novelty checking
3. LLM calls SemanticScholarSearchTool
4. Results inform idea refinement
5. Process repeats for `num_reflections` rounds

---

### 10. Metric System

**Location**: [`ai_scientist/treesearch/utils/metric.py`](../ai_scientist/treesearch/utils/metric.py)

**Purpose**: Track performance metrics across experiments and datasets.

#### `MetricValue` Class

**Attributes**:
- `value`: Primary metric value (float or None)
- `metrics_dict`: Dictionary of all metrics
- `dataset`: Dataset name
- `maximize`: Whether higher is better (boolean)

**Methods**:
- `is_better_than(other)`: Compare two MetricValue objects
- `to_dict()`: Serialize to dictionary
- `from_dict(data)`: Deserialize from dictionary

**Multi-Dataset Support**:
MetricValue can track metrics across multiple datasets:
```python
metric = MetricValue(
    value=0.95,  # Primary metric
    metrics_dict={
        "dataset1": {"accuracy": 0.95, "f1": 0.93},
        "dataset2": {"accuracy": 0.87, "f1": 0.85}
    },
    maximize=True
)
```

---

### 11. Utility Modules

**Location**: [`ai_scientist/treesearch/utils/`](../ai_scientist/treesearch/utils/)

#### Response Utilities (`response.py`)

**Key Functions**:
- `extract_code(text)`: Extract Python code from markdown blocks
- `trim_long_string(s, max_len)`: Truncate strings for logging

#### Tree Export (`tree_export.py`)

**Purpose**: Generate interactive HTML visualization of solution tree.

**Function**: `export_tree_to_html(journal, output_path)`
- Creates `unified_tree_viz.html`
- Interactive tree with node details
- Metrics, code, and plots for each node

#### Serialization (`serialize.py`)

**Purpose**: JSON serialization for complex objects.

**Functions**:
- `serialize_node(node)`: Node → JSON
- `deserialize_node(data)`: JSON → Node
- Handles circular references in tree structure

#### Token Tracking (`token_tracker.py`)

**Purpose**: Track LLM API usage and costs.

**Features**:
- Decorator-based tracking
- Per-model token counting
- Cost estimation
- Usage reports

---

## Complete Data Flow

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. IDEATION PHASE                                           │
├─────────────────────────────────────────────────────────────┤
│ Input: my_research_topic.md                                 │
│   - Title, Keywords, Abstract, TL;DR                        │
│                                                             │
│ Process: perform_ideation_temp_free.py                      │
│   ├─ LLM generates research ideas                          │
│   ├─ SemanticScholarSearchTool: novelty check             │
│   └─ Iterative refinement (num_reflections rounds)         │
│                                                             │
│ Output: my_research_topic.json                              │
│   - Hypothesis                                              │
│   - Proposed experiments                                    │
│   - Related work analysis                                   │
│   - Risk factors                                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. EXPERIMENT PHASE                                         │
├─────────────────────────────────────────────────────────────┤
│ Input: my_research_topic.json + bfts_config.yaml           │
│                                                             │
│ Process: launch_scientist_bfts.py                           │
│   ↓                                                         │
│   AgentManager.run()                                        │
│   │                                                         │
│   ├─ STAGE 1: Initial Implementation                       │
│   │   ├─ ParallelAgent creates initial drafts              │
│   │   ├─ Tree search: draft → improve → debug              │
│   │   ├─ Code execution via Interpreter                    │
│   │   ├─ Metric extraction via metric_parse_spec           │
│   │   ├─ Plot analysis via vlm_feedback_spec               │
│   │   └─ Journal.get_best_node() selects winner            │
│   │                                                         │
│   ├─ STAGE 2: Baseline Tuning                              │
│   │   [Same process with hyperparameter focus]             │
│   │                                                         │
│   ├─ STAGE 3: Creative Research                            │
│   │   [Same process with novel ideas]                      │
│   │                                                         │
│   └─ STAGE 4: Ablation Studies                             │
│       [Same process with component analysis]                │
│                                                             │
│ Output: experiments/timestamp_ideaname/                     │
│   - logs/0-run/unified_tree_viz.html                        │
│   - best_implementation.py                                  │
│   - plots/                                                  │
│   - metrics.json                                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. WRITING PHASE                                            │
├─────────────────────────────────────────────────────────────┤
│ Input: Experimental results + plots                         │
│                                                             │
│ Process:                                                    │
│   ├─ aggregate_plots(): Select best visualizations         │
│   │                                                         │
│   ├─ perform_writeup() / perform_icbinb_writeup()          │
│   │   ├─ Generate Abstract                                 │
│   │   ├─ Generate Introduction                             │
│   │   ├─ Generate Methods                                  │
│   │   ├─ Generate Results                                  │
│   │   ├─ Generate Discussion                               │
│   │   ├─ Generate Conclusion                               │
│   │   │                                                     │
│   │   ├─ gather_citations() [iterative]                    │
│   │   │   ├─ LLM identifies claims needing citations       │
│   │   │   ├─ SemanticScholar search                        │
│   │   │   ├─ LLM selects best papers                       │
│   │   │   └─ Insert citations + bibliography               │
│   │   │                                                     │
│   │   └─ compile_latex() → PDF                             │
│   │                                                         │
│   └─ Output: timestamp_ideaname.pdf                         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. REVIEW PHASE                                             │
├─────────────────────────────────────────────────────────────┤
│ Input: Generated PDF                                        │
│                                                             │
│ Process:                                                    │
│   ├─ perform_review(): Text-based review                   │
│   │   - Novelty, significance                              │
│   │   - Technical correctness                              │
│   │   - Clarity, organization                              │
│   │   - Experimental rigor                                 │
│   │                                                         │
│   └─ perform_imgs_cap_ref_review(): Visual review          │
│       - Figure quality                                      │
│       - Caption appropriateness                             │
│       - Reference formatting                                │
│                                                             │
│ Output: review.json + vlm_review.json                       │
└─────────────────────────────────────────────────────────────┘
```

### Tree Search Detail (Within Each Stage)

```
ParallelAgent.step() Iteration:

1. Node Selection
   └─ Select best leaf nodes (up to num_workers)
      - Non-buggy nodes ranked by metric
      - Buggy nodes if debug_prob triggers

2. Parallel Expansion (num_workers threads)
   ├─ Thread 1: Node A
   │   ├─ Is buggy? → Debug (if debug_depth < max_debug_depth)
   │   └─ Not buggy? → Improve
   │
   ├─ Thread 2: Node B
   │   └─ [Same logic]
   │
   └─ Thread N: Node N
       └─ [Same logic]

3. Code Generation (per thread)
   ├─ Build prompt with:
   │   - Current code
   │   - Metrics & analysis
   │   - Error messages (if debugging)
   │   - Improvement suggestions (if improving)
   │
   ├─ LLM generates new code
   └─ Create child node

4. Execution (per new node)
   ├─ Interpreter.run(code, timeout)
   ├─ Capture stdout/stderr
   └─ Set is_buggy if exception

5. Evaluation (per successful node)
   ├─ Extract metrics via metric_parse_spec
   ├─ Generate plots
   ├─ VLM analysis via vlm_feedback_spec
   └─ LLM analysis of results

6. Journal Update
   └─ Add all new nodes to Journal

Repeat until 'steps' iterations completed
```

---

## Key Design Patterns

### 1. Hierarchical Agent Architecture

**Pattern**: Nested agents with clear responsibilities.

```
AgentManager (Orchestration)
  ↓ creates
ParallelAgent (Execution)
  ↓ extends
MinimalAgent (Prompting)
```

**Benefits**:
- Clear separation of concerns
- Easier testing and debugging
- Modular improvements

### 2. Tree Search with Journaling

**Pattern**: Solution space explored as tree, complete history maintained.

**Components**:
- `Node`: Atomic unit of exploration
- `Journal`: Tree container with analytics
- `ParallelAgent`: Tree expansion logic

**Benefits**:
- Full experimental history preserved
- Easy rollback to previous solutions
- Comparative analysis across branches
- Visualization of exploration process

### 3. Multi-Stage Progression

**Pattern**: Structured progression through research phases.

**Implementation**:
- Each stage has clear goals
- Best result from stage N → starting point for stage N+1
- LLM-based stage completion checks
- Dynamic sub-stage generation

**Benefits**:
- Mimics real research workflow
- Prevents premature optimization
- Natural complexity progression

### 4. Tool-Based Extensibility

**Pattern**: Abstract base class for capabilities.

```python
class BaseTool(ABC):
    @abstractmethod
    def use_tool(self, **kwargs):
        pass
```

**Benefits**:
- Easy addition of new tools
- Consistent interface for LLM
- Testable in isolation

### 5. Backend Abstraction

**Pattern**: Unified interface across LLM providers.

**Implementation**:
- `create_client(model)`: Returns appropriate client
- `query()`: Common calling convention
- `FunctionSpec`: Provider-agnostic structured outputs

**Benefits**:
- Switch providers without code changes
- Cost optimization through model selection
- Fallback strategies

### 6. Structured Outputs

**Pattern**: JSON schemas for reliable parsing.

**Implementation**: `FunctionSpec` with JSON Schema

**Benefits**:
- Reliable extraction of structured data
- Type safety
- Validation at API level

### 7. Safety-First Execution

**Pattern**: Sandboxed execution of untrusted code.

**Implementation**: `Interpreter` with:
- Process isolation
- Timeout enforcement
- Output capture
- Exception handling

**Benefits**:
- Prevents system damage
- Handles infinite loops
- Logs all behavior

---

## Important Files Reference

### Entry Points

| File | Purpose |
|------|---------|
| [`launch_scientist_bfts.py`](../launch_scientist_bfts.py) | Main orchestrator for experiments |
| [`ai_scientist/perform_ideation_temp_free.py`](../ai_scientist/perform_ideation_temp_free.py) | Research idea generation |

### Core Logic

| File | Purpose |
|------|---------|
| [`ai_scientist/treesearch/agent_manager.py`](../ai_scientist/treesearch/agent_manager.py) | Multi-stage workflow orchestration |
| [`ai_scientist/treesearch/parallel_agent.py`](../ai_scientist/treesearch/parallel_agent.py) | Tree search implementation |
| [`ai_scientist/treesearch/journal.py`](../ai_scientist/treesearch/journal.py) | Solution tree data structure |
| [`ai_scientist/treesearch/interpreter.py`](../ai_scientist/treesearch/interpreter.py) | Safe code execution |

### LLM Integration

| File | Purpose |
|------|---------|
| [`ai_scientist/llm.py`](../ai_scientist/llm.py) | Unified LLM interface |
| [`ai_scientist/treesearch/backend/backend_openai.py`](../ai_scientist/treesearch/backend/backend_openai.py) | OpenAI integration |
| [`ai_scientist/treesearch/backend/backend_anthropic.py`](../ai_scientist/treesearch/backend/backend_anthropic.py) | Anthropic integration |
| [`ai_scientist/treesearch/backend/utils.py`](../ai_scientist/treesearch/backend/utils.py) | Backend utilities & FunctionSpec |

### Paper Generation

| File | Purpose |
|------|---------|
| [`ai_scientist/perform_writeup.py`](../ai_scientist/perform_writeup.py) | Paper writing (8-page) |
| [`ai_scientist/perform_icbinb_writeup.py`](../ai_scientist/perform_icbinb_writeup.py) | Workshop paper (4-page) |
| [`ai_scientist/perform_plotting.py`](../ai_scientist/perform_plotting.py) | Plot aggregation |
| [`ai_scientist/perform_llm_review.py`](../ai_scientist/perform_llm_review.py) | Text-based review |
| [`ai_scientist/perform_vlm_review.py`](../ai_scientist/perform_vlm_review.py) | Visual review |

### Configuration

| File | Purpose |
|------|---------|
| [`bfts_config.yaml`](../bfts_config.yaml) | Tree search configuration |
| [`ai_scientist/treesearch/utils/config.py`](../ai_scientist/treesearch/utils/config.py) | Configuration classes |

### Tools & Utilities

| File | Purpose |
|------|---------|
| [`ai_scientist/tools/base_tool.py`](../ai_scientist/tools/base_tool.py) | Tool base class |
| [`ai_scientist/tools/semantic_scholar_search.py`](../ai_scientist/tools/semantic_scholar_search.py) | Literature search |
| [`ai_scientist/treesearch/utils/metric.py`](../ai_scientist/treesearch/utils/metric.py) | Metric tracking |
| [`ai_scientist/treesearch/utils/response.py`](../ai_scientist/treesearch/utils/response.py) | Response parsing |
| [`ai_scientist/treesearch/utils/tree_export.py`](../ai_scientist/treesearch/utils/tree_export.py) | Tree visualization |
| [`ai_scientist/treesearch/utils/token_tracker.py`](../ai_scientist/treesearch/utils/token_tracker.py) | API usage tracking |

---

## Configuration Guide

### Quick Start Configuration

Minimal `bfts_config.yaml` for testing:

```yaml
agent:
  steps: 3                    # Fewer iterations
  num_workers: 2              # Less parallelism
  num_seeds: 1                # Single seed

  code:
    model: gpt-4o-2024-11-20  # Faster than Claude
    temp: 0.8
    max_tokens: 8000

  search:
    num_drafts: 2             # Fewer initial attempts
    max_debug_depth: 2
    debug_prob: 0.5

exec:
  timeout: 1800               # 30 minutes
```

### Production Configuration

Optimized for quality:

```yaml
agent:
  steps: 10                   # More exploration
  num_workers: 4              # High parallelism
  num_seeds: 3                # Multiple seeds for validation

  code:
    model: claude-3-5-sonnet-20241022  # Best performance
    temp: 1.0
    max_tokens: 12000

  feedback:
    model: gpt-4o-2024-11-20
    temp: 0.5

  vlm_feedback:
    model: gpt-4o-2024-11-20
    temp: 0.3                 # Lower for consistency

  search:
    num_drafts: 4             # More diverse starts
    max_debug_depth: 3
    debug_prob: 0.6           # Higher debugging priority

exec:
  timeout: 3600               # 1 hour per experiment
```

### Cost Optimization

Balance performance and cost:

```yaml
agent:
  code:
    model: gpt-4o-2024-11-20  # Cheaper than Claude

  feedback:
    model: gpt-4o-mini         # Much cheaper for analysis

  vlm_feedback:
    model: gpt-4o-2024-11-20  # VLM required
```

---

## Extending the System

### Adding a New Tool

1. **Create tool class**:

```python
# ai_scientist/tools/my_new_tool.py

from ai_scientist.tools.base_tool import BaseTool

class MyNewTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_new_tool",
            description="What this tool does",
            parameters={
                "param1": "description",
                "param2": "description"
            }
        )

    def use_tool(self, param1, param2):
        # Implementation
        result = do_something(param1, param2)
        return result
```

2. **Register in ideation**:

```python
# ai_scientist/perform_ideation_temp_free.py

from ai_scientist.tools.my_new_tool import MyNewTool

tools = [
    SemanticScholarSearchTool(),
    MyNewTool()  # Add here
]
```

### Adding a New LLM Provider

1. **Create backend file**:

```python
# ai_scientist/treesearch/backend/backend_mynewprovider.py

def query_mynewprovider(messages, model, func_spec=None, **kwargs):
    # Implementation
    pass
```

2. **Update backend utils**:

```python
# ai_scientist/treesearch/backend/utils.py

def query(messages, model, **kwargs):
    if "mynewprovider" in model:
        return query_mynewprovider(messages, model, **kwargs)
    # ... existing providers
```

3. **Update client creation**:

```python
# ai_scientist/llm.py

def create_client(model):
    if "mynewprovider" in model:
        return MyNewProviderClient()
    # ... existing providers
```

### Customizing Stage Progression

Modify stage definitions in `agent_manager.py`:

```python
stages = [
    Stage(
        name="my_custom_stage",
        description="Custom research phase",
        goals=[
            "Goal 1",
            "Goal 2"
        ],
        max_iterations=5,
        num_drafts=3,
        stage_number=5  # New stage
    )
]
```

---

## Troubleshooting

### Common Issues

**Issue**: "CUDA Out of Memory"
- **Solution**: Update ideation prompt to suggest smaller models
- **Config**: Reduce `num_workers` in `bfts_config.yaml`

**Issue**: "Semantic Scholar rate limit"
- **Solution**: Set `S2_API_KEY` environment variable
- **Alternative**: Skip citation phase with appropriate flags

**Issue**: "Timeout during execution"
- **Solution**: Increase `exec.timeout` in `bfts_config.yaml`
- **Check**: Code may have inefficiencies (infinite loops, etc.)

**Issue**: "Low success rate"
- **Solution**: Use Claude 3.5 Sonnet for `agent.code.model`
- **Increase**: `agent.search.num_drafts` for more initial attempts
- **Increase**: `agent.steps` for more exploration

**Issue**: "PDF not generated"
- **Check**: LaTeX compilation errors in logs
- **Ensure**: `poppler` and `chktex` installed
- **Review**: Generated LaTeX for syntax errors

---

## Performance Characteristics

### Typical Runtime (per experiment)

| Phase | Duration | Cost (USD) |
|-------|----------|------------|
| Ideation | 5-10 min | $2-5 |
| Stage 1 | 30-60 min | $5-10 |
| Stage 2 | 30-60 min | $5-10 |
| Stage 3 | 30-60 min | $5-10 |
| Stage 4 | 30-60 min | $5-10 |
| Writing | 20-30 min | $5 |
| Review | 5-10 min | $1-2 |
| **Total** | **3-5 hours** | **$25-50** |

*Using Claude 3.5 Sonnet for experiments, GPT-4o for writing*

### Scaling Factors

**Increases runtime/cost**:
- Higher `steps` (more iterations)
- Higher `num_workers` (more parallel paths)
- Higher `num_drafts` (more initial attempts)
- Complex experiments (slow execution)
- Multiple datasets

**Decreases runtime/cost**:
- Simpler experiments
- Faster models (GPT-4o vs Claude)
- Lower temperature (less creative, faster convergence)
- Efficient code generation

---

## Summary

The AI Scientist-v2 implements a **complete automated research pipeline** through:

1. **Agentic Tree Search**: Intelligent exploration of solution spaces
2. **Multi-Stage Progression**: Natural research workflow from implementation to ablation
3. **Parallel Exploration**: Efficient use of resources via multiple workers
4. **LLM-Based Evaluation**: Soft metrics for code quality and plot analysis
5. **Structured Outputs**: Reliable data extraction via function calling
6. **Safety Sandboxing**: Secure execution of LLM-generated code
7. **Full Paper Writing**: Automated LaTeX generation with citations and reviews

The system is designed to be:
- **Model-agnostic**: Works with OpenAI, Anthropic, Google, Ollama
- **Field-agnostic**: Adaptable across ML domains via ideation
- **Extensible**: Easy addition of tools, models, and stages
- **Safe**: Sandboxed execution prevents system damage

This architecture enables fully autonomous scientific research from idea generation through paper publication, with sophisticated exploration strategies and quality control mechanisms.
