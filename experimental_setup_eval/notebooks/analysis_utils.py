"""
Utility functions for experimental setup evaluation analysis notebooks.

This module provides reusable functions for:
- Loading and validating run data
- Extracting quality scores and metadata
- Computing statistics and comparisons
- Creating visualizations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Visualization settings
FIGURE_SIZE_LARGE = (14, 8)
FIGURE_SIZE_WIDE = (16, 7)
FIGURE_SIZE_STANDARD = (12, 6)

# Analysis thresholds
IMPROVEMENT_THRESHOLD = 0.1  # Minimum improvement to highlight
SIGNIFICANT_IMPROVEMENT_PCT = 10.0  # Minimum % improvement to highlight

# Approach names
APPROACH_NON_AGENTIC = 'non_agentic'
APPROACH_AGENTIC = 'agentic'


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file with error handling.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data or None if error
    """
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  File not found: {path}")
        return None
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON decode error in {path}: {e}")
        return None


def discover_paper_folders(run_path: Path) -> List[Path]:
    """Discover all paper folders in the run directory.

    Args:
        run_path: Path to run directory

    Returns:
        List of paper folder paths
    """
    paper_folders = []
    for item in run_path.iterdir():
        if item.is_dir() and (item / "results.json").exists():
            paper_folders.append(item)
    return sorted(paper_folders)


def load_run_data(run_folder: str, specific_papers: List[str] = None) -> Tuple[Optional[Dict], Dict, Optional[Path]]:
    """Load all data from a run folder.

    Args:
        run_folder: Path to run folder
        specific_papers: Optional list of specific paper IDs to load

    Returns:
        Tuple of (summary_data, papers_data, base_path)
    """
    base_path = Path(run_folder)

    if not base_path.exists():
        print(f"❌ Run folder does not exist: {base_path}")
        return None, {}, None

    # Load summary
    summary_data = load_json(base_path / "summary.json")

    # Discover paper folders
    paper_folders = discover_paper_folders(base_path)

    # Load all paper results
    papers_data = {}
    for paper_folder in paper_folders:
        paper_id = paper_folder.name

        # Skip if filtering to specific papers
        if specific_papers and paper_id not in specific_papers:
            continue

        results = load_json(paper_folder / "results.json")
        context = load_json(paper_folder / "paper_context.json")

        if results:
            papers_data[paper_id] = {
                'results': results,
                'context': context,
                'folder': paper_folder
            }

    return summary_data, papers_data, base_path


# ============================================================================
# DATA EXTRACTION FUNCTIONS
# ============================================================================

def get_quality_score(results: Dict, approach: str = APPROACH_NON_AGENTIC) -> float:
    """Safely extract quality score for an approach.

    Args:
        results: Results dictionary
        approach: Either 'non_agentic' or 'agentic'

    Returns:
        Quality score (0.0 if not found)
    """
    return results.get(approach, {}).get('quality', {}).get('overall_score', 0.0)


def get_component_scores(results: Dict, approach: str = APPROACH_NON_AGENTIC) -> Dict[str, float]:
    """Safely extract component scores for an approach.

    Args:
        results: Results dictionary
        approach: Either 'non_agentic' or 'agentic'

    Returns:
        Dictionary of component scores
    """
    return results.get(approach, {}).get('quality', {}).get('component_scores', {})


def get_metadata(results: Dict, approach: str = APPROACH_NON_AGENTIC) -> Dict:
    """Safely extract metadata for an approach.

    Args:
        results: Results dictionary
        approach: Either 'non_agentic' or 'agentic'

    Returns:
        Metadata dictionary
    """
    return results.get(approach, {}).get('setup', {}).get('metadata', {})


def get_exploration_tree(results: Dict) -> List[Dict]:
    """Extract agentic exploration tree.

    Args:
        results: Results dictionary

    Returns:
        List of draft dictionaries
    """
    return results.get(APPROACH_AGENTIC, {}).get('setup', {}).get('exploration_tree', [])


def get_paper_context(context: Dict) -> Dict[str, str]:
    """Extract key paper context fields.

    Args:
        context: Context dictionary

    Returns:
        Dictionary with title, domain, research_question
    """
    if not context:
        return {
            'title': 'Unknown',
            'domain': 'Unknown',
            'research_question': 'N/A'
        }

    return {
        'title': context.get('title', 'Unknown'),
        'domain': context.get('domain', 'Unknown'),
        'research_question': context.get('research_question', 'N/A')
    }


# ============================================================================
# DATA AGGREGATION FUNCTIONS
# ============================================================================

def aggregate_component_scores(papers_data: Dict) -> pd.DataFrame:
    """Aggregate component scores across all papers.

    Args:
        papers_data: Dictionary of paper data

    Returns:
        DataFrame with mean component scores and improvements
    """
    all_components = {}

    for paper_id, paper_info in papers_data.items():
        results = paper_info['results']

        non_agentic_scores = get_component_scores(results, APPROACH_NON_AGENTIC)
        agentic_scores = get_component_scores(results, APPROACH_AGENTIC)

        for component in non_agentic_scores.keys():
            if component not in all_components:
                all_components[component] = {
                    APPROACH_NON_AGENTIC: [],
                    APPROACH_AGENTIC: []
                }

            all_components[component][APPROACH_NON_AGENTIC].append(
                non_agentic_scores.get(component, 0)
            )
            all_components[component][APPROACH_AGENTIC].append(
                agentic_scores.get(component, 0)
            )

    # Calculate means
    component_means = []
    for component, scores in all_components.items():
        non_mean = np.mean(scores[APPROACH_NON_AGENTIC]) if scores[APPROACH_NON_AGENTIC] else 0
        agentic_mean = np.mean(scores[APPROACH_AGENTIC]) if scores[APPROACH_AGENTIC] else 0
        improvement = agentic_mean - non_mean
        improvement_pct = (improvement / non_mean * 100) if non_mean > 0 else 0

        component_means.append({
            'Component': component,
            'Non-Agentic': non_mean,
            'Agentic': agentic_mean,
            'Improvement': improvement,
            'Improvement %': improvement_pct
        })

    df = pd.DataFrame(component_means)
    return df.sort_values('Improvement', ascending=False)


def aggregate_cost_benefit(papers_data: Dict) -> pd.DataFrame:
    """Aggregate cost-benefit data across all papers.

    Args:
        papers_data: Dictionary of paper data

    Returns:
        DataFrame with cost and quality metrics
    """
    cost_benefit_data = []

    for paper_id, paper_info in papers_data.items():
        results = paper_info['results']

        non_agentic_meta = get_metadata(results, APPROACH_NON_AGENTIC)
        agentic_meta = get_metadata(results, APPROACH_AGENTIC)

        non_quality = get_quality_score(results, APPROACH_NON_AGENTIC)
        agentic_quality = get_quality_score(results, APPROACH_AGENTIC)

        if non_agentic_meta and agentic_meta:
            cost_benefit_data.append({
                'paper_id': paper_id,
                'non_agentic_time': non_agentic_meta.get('time_elapsed', 0),
                'agentic_time': agentic_meta.get('time_elapsed', 0),
                'non_agentic_calls': non_agentic_meta.get('llm_calls', 0),
                'agentic_calls': agentic_meta.get('llm_calls', 0),
                'non_agentic_quality': non_quality,
                'agentic_quality': agentic_quality
            })

    return pd.DataFrame(cost_benefit_data)


def compute_cost_benefit_summary(df_cost: pd.DataFrame) -> Dict:
    """Compute summary statistics from cost-benefit data.

    Args:
        df_cost: Cost-benefit DataFrame

    Returns:
        Dictionary of summary statistics
    """
    if df_cost.empty:
        return {}

    total_non_time = df_cost['non_agentic_time'].sum()
    total_agentic_time = df_cost['agentic_time'].sum()
    total_non_calls = df_cost['non_agentic_calls'].sum()
    total_agentic_calls = df_cost['agentic_calls'].sum()

    avg_non_quality = df_cost['non_agentic_quality'].mean()
    avg_agentic_quality = df_cost['agentic_quality'].mean()

    time_ratio = total_agentic_time / total_non_time if total_non_time > 0 else 0
    calls_ratio = total_agentic_calls / total_non_calls if total_non_calls > 0 else 0
    quality_gain_pct = ((avg_agentic_quality - avg_non_quality) / avg_non_quality * 100) if avg_non_quality > 0 else 0

    num_papers = len(df_cost)

    return {
        'total_non_time': total_non_time,
        'total_agentic_time': total_agentic_time,
        'avg_non_time': total_non_time / num_papers,
        'avg_agentic_time': total_agentic_time / num_papers,
        'total_non_calls': total_non_calls,
        'total_agentic_calls': total_agentic_calls,
        'avg_non_calls': total_non_calls / num_papers,
        'avg_agentic_calls': total_agentic_calls / num_papers,
        'avg_non_quality': avg_non_quality,
        'avg_agentic_quality': avg_agentic_quality,
        'time_ratio': time_ratio,
        'calls_ratio': calls_ratio,
        'quality_gain_pct': quality_gain_pct,
        'num_papers': num_papers
    }


def create_paper_comparison_df(papers_data: Dict) -> pd.DataFrame:
    """Create a comparison DataFrame for all papers.

    Args:
        papers_data: Dictionary of paper data

    Returns:
        DataFrame with per-paper comparison
    """
    comparison_data = []

    for paper_id, paper_info in papers_data.items():
        results = paper_info['results']
        context = paper_info.get('context', {})

        paper_context = get_paper_context(context)
        non_quality = get_quality_score(results, APPROACH_NON_AGENTIC)
        agentic_quality = get_quality_score(results, APPROACH_AGENTIC)

        comparison_data.append({
            'Paper ID': paper_id[:40] + '...' if len(paper_id) > 40 else paper_id,
            'Title': paper_context['title'][:60] + '...' if len(paper_context['title']) > 60 else paper_context['title'],
            'Domain': paper_context['domain'],
            'Non-Agentic': non_quality,
            'Agentic': agentic_quality,
            'Improvement': agentic_quality - non_quality,
            'Improvement %': ((agentic_quality - non_quality) / non_quality * 100) if non_quality > 0 else 0
        })

    return pd.DataFrame(comparison_data)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def setup_plot_style():
    """Configure matplotlib/seaborn styling for publication-quality figures."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.figsize"] = FIGURE_SIZE_STANDARD
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14


def plot_component_comparison(df_components: pd.DataFrame, title: str = "Component Scores Comparison") -> plt.Figure:
    """Create grouped bar chart comparing component scores.

    Args:
        df_components: DataFrame with component scores
        title: Plot title

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)

    x = np.arange(len(df_components))
    width = 0.35

    bars1 = ax.bar(x - width/2, df_components['Non-Agentic'], width,
                   label='Non-Agentic', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, df_components['Agentic'], width,
                   label='Agentic', color='#e74c3c', alpha=0.8, edgecolor='black')

    ax.set_xlabel('Quality Dimensions', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mean Score (0-1)', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df_components['Component'], rotation=45, ha='right')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    # Add improvement markers for significant gains
    for i, (idx, row) in enumerate(df_components.iterrows()):
        if row['Improvement'] > IMPROVEMENT_THRESHOLD:
            ax.annotate(f"+{row['Improvement %']:.0f}%",
                       xy=(i, max(row['Non-Agentic'], row['Agentic']) + 0.02),
                       ha='center', fontsize=9, color='green', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_cost_benefit_scatter(df_cost: pd.DataFrame) -> plt.Figure:
    """Create scatter plots showing time/calls vs quality trade-offs.

    Args:
        df_cost: Cost-benefit DataFrame

    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)

    # Left: Time vs Quality
    ax1.scatter(df_cost['non_agentic_time'], df_cost['non_agentic_quality'],
               s=200, c='#3498db', alpha=0.7, edgecolors='black', linewidth=2,
               label='Non-Agentic', marker='o')
    ax1.scatter(df_cost['agentic_time'], df_cost['agentic_quality'],
               s=200, c='#e74c3c', alpha=0.7, edgecolors='black', linewidth=2,
               label='Agentic', marker='s')

    # Draw connecting lines
    for _, row in df_cost.iterrows():
        ax1.plot([row['non_agentic_time'], row['agentic_time']],
                [row['non_agentic_quality'], row['agentic_quality']],
                'gray', alpha=0.3, linestyle='--', linewidth=1)

    ax1.set_xlabel('Time (seconds)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Quality Score', fontweight='bold', fontsize=12)
    ax1.set_title('Time vs Quality Trade-off', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right: LLM Calls vs Quality
    ax2.scatter(df_cost['non_agentic_calls'], df_cost['non_agentic_quality'],
               s=200, c='#3498db', alpha=0.7, edgecolors='black', linewidth=2,
               label='Non-Agentic', marker='o')
    ax2.scatter(df_cost['agentic_calls'], df_cost['agentic_quality'],
               s=200, c='#e74c3c', alpha=0.7, edgecolors='black', linewidth=2,
               label='Agentic', marker='s')

    # Draw connecting lines
    for _, row in df_cost.iterrows():
        ax2.plot([row['non_agentic_calls'], row['agentic_calls']],
                [row['non_agentic_quality'], row['agentic_quality']],
                'gray', alpha=0.3, linestyle='--', linewidth=1)

    ax2.set_xlabel('LLM API Calls', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Quality Score', fontweight='bold', fontsize=12)
    ax2.set_title('API Calls vs Quality Trade-off', fontweight='bold', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# FORMATTING FUNCTIONS
# ============================================================================

def format_exploration_tree(exploration_tree: List[Dict]) -> pd.DataFrame:
    """Format exploration tree into a readable DataFrame.

    Args:
        exploration_tree: List of draft dictionaries

    Returns:
        DataFrame with draft summary
    """
    if not exploration_tree:
        return pd.DataFrame()

    draft_summary = []
    for draft in exploration_tree:
        draft_summary.append({
            'Draft': draft.get('draft_id', 'Unknown'),
            'Score': f"{draft.get('score', 0):.4f}",
            'Best': '✓' if draft.get('is_best', False) else '',
            'Refinement': '✓' if draft.get('is_refinement', False) else ''
        })

    return pd.DataFrame(draft_summary)


def create_cost_summary_table(summary_stats: Dict) -> pd.DataFrame:
    """Create formatted cost summary table.

    Args:
        summary_stats: Summary statistics dictionary

    Returns:
        DataFrame with formatted metrics
    """
    if not summary_stats:
        return pd.DataFrame()

    num_papers = summary_stats['num_papers']

    return pd.DataFrame({
        'Metric': [
            'Total Time (s)',
            'Avg Time per Paper (s)',
            'Total LLM Calls',
            'Avg LLM Calls per Paper',
            'Avg Quality Score',
            'Quality per Second',
            'Quality per Call'
        ],
        'Non-Agentic': [
            f"{summary_stats['total_non_time']:.1f}",
            f"{summary_stats['avg_non_time']:.1f}",
            f"{summary_stats['total_non_calls']:.0f}",
            f"{summary_stats['avg_non_calls']:.1f}",
            f"{summary_stats['avg_non_quality']:.3f}",
            f"{summary_stats['avg_non_quality']/summary_stats['avg_non_time']:.4f}",
            f"{summary_stats['avg_non_quality']/summary_stats['avg_non_calls']:.3f}"
        ],
        'Agentic': [
            f"{summary_stats['total_agentic_time']:.1f}",
            f"{summary_stats['avg_agentic_time']:.1f}",
            f"{summary_stats['total_agentic_calls']:.0f}",
            f"{summary_stats['avg_agentic_calls']:.1f}",
            f"{summary_stats['avg_agentic_quality']:.3f}",
            f"{summary_stats['avg_agentic_quality']/summary_stats['avg_agentic_time']:.4f}",
            f"{summary_stats['avg_agentic_quality']/summary_stats['avg_agentic_calls']:.3f}"
        ],
        'Ratio (A/NA)': [
            f"{summary_stats['time_ratio']:.2f}x",
            f"{summary_stats['time_ratio']:.2f}x",
            f"{summary_stats['calls_ratio']:.1f}x",
            f"{summary_stats['calls_ratio']:.1f}x",
            f"+{summary_stats['quality_gain_pct']:.1f}%",
            f"{(summary_stats['avg_agentic_quality']/summary_stats['avg_agentic_time'])/(summary_stats['avg_non_quality']/summary_stats['avg_non_time']):.2f}x",
            f"{(summary_stats['avg_agentic_quality']/summary_stats['avg_agentic_calls'])/(summary_stats['avg_non_quality']/summary_stats['avg_non_calls']):.2f}x"
        ]
    })


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def print_section_header(title: str, width: int = 100, char: str = '='):
    """Print a formatted section header.

    Args:
        title: Section title
        width: Width of header line
        char: Character to use for border
    """
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}\n")


def print_paper_header(paper_id: str, idx: int, total: int):
    """Print a formatted paper header.

    Args:
        paper_id: Paper identifier
        idx: Current paper index
        total: Total number of papers
    """
    print(f"\n{'#' * 100}")
    print(f"PAPER {idx}/{total}: {paper_id}")
    print(f"{'#' * 100}")


def print_load_summary(summary_data: Dict, papers_data: Dict):
    """Print summary of loaded data.

    Args:
        summary_data: Summary dictionary
        papers_data: Papers data dictionary
    """
    print_section_header("LOADING RUN DATA", width=80)

    if summary_data:
        print(f"✓ Summary loaded: {summary_data.get('timestamp', 'unknown')}")
        print(f"  - Total papers: {summary_data.get('total_papers', 0)}")
        print(f"  - Successful: {summary_data.get('successful', 0)}")
        print(f"  - Failed: {summary_data.get('failed', 0)}")
        print(f"  - Success rate: {summary_data.get('success_rate', 0)*100:.1f}%")
        perf = summary_data.get('performance', {})
        print(f"  - Total time: {perf.get('total_time_seconds', 0):.1f}s")

    if papers_data:
        print(f"\n✓ Loaded {len(papers_data)} paper(s):")
        for i, (paper_id, paper_info) in enumerate(list(papers_data.items())[:5]):
            context = paper_info.get('context', {})
            title = context.get('title', 'Unknown')[:60] if context else 'Unknown'
            winner = paper_info.get('results', {}).get('winner', 'N/A')
            print(f"  • {paper_id[:40]}... (Winner: {winner})")

        if len(papers_data) > 5:
            print(f"  ... and {len(papers_data) - 5} more")
    else:
        print("❌ No papers loaded")

    print("=" * 80 + "\n")


# ============================================================================
# SINGLE-PAPER VISUALIZATION FUNCTIONS
# ============================================================================

def extract_setup_components(setup_data: Dict) -> Dict:
    """Extract baselines, metrics, and datasets from setup data.

    Args:
        setup_data: Setup dictionary (from results.json)

    Returns:
        Dictionary with baselines, metrics, datasets lists
    """
    # Handle nested structure - setup might be in 'setup' -> 'setup' (double nesting)
    # or 'setup' -> 'ai_setup' (old structure)
    if not setup_data:
        return {'baselines': [], 'metrics': [], 'datasets': []}

    setup = setup_data.get('setup', {})

    # Check for double-nested 'setup' structure (current format)
    if 'setup' in setup and isinstance(setup['setup'], dict):
        setup = setup['setup']
    # Check for old 'ai_setup' structure (backward compatibility)
    elif 'ai_setup' in setup:
        setup = setup['ai_setup']

    return {
        'baselines': setup.get('baselines', []),
        'metrics': setup.get('metrics', []),
        'datasets': setup.get('datasets', [])
    }


def format_component_list(components: List, component_type: str = "item") -> str:
    """Format a list of components (baselines/metrics/datasets) for display.

    Args:
        components: List of components (can be strings or dicts)
        component_type: Type of component for display

    Returns:
        Formatted string
    """
    if not components:
        return f"  (No {component_type}s found)"

    result = []
    for i, comp in enumerate(components, 1):
        if isinstance(comp, dict):
            name = comp.get('name', 'Unknown')
            desc = comp.get('description', 'No description')
            result.append(f"  {i}. {name}")
            result.append(f"     {desc}")
        else:
            result.append(f"  {i}. {comp}")

    return "\n".join(result)


def get_alignment_data(results: Dict) -> Dict:
    """Extract alignment data for both approaches.

    Args:
        results: Results dictionary

    Returns:
        Dictionary with alignment info for both approaches
    """
    non_agentic_align = results.get(APPROACH_NON_AGENTIC, {}).get('alignment', {})
    agentic_align = results.get(APPROACH_AGENTIC, {}).get('alignment', {})

    def extract_component_alignment(align_data: Dict) -> Dict:
        """Extract component alignment scores from component_scores."""
        component_scores = align_data.get('component_scores', {})

        return {
            'overall': align_data.get('overall_alignment', 0),
            'baselines': {
                'recall': component_scores.get('baseline_recall', 0),
                'precision': component_scores.get('baseline_precision', 0),
                'f1': component_scores.get('baseline_f1', 0)
            },
            'metrics': {
                'recall': component_scores.get('metric_recall', 0),
                'precision': component_scores.get('metric_precision', 0),
                'f1': component_scores.get('metric_f1', 0)
            },
            'datasets': {
                'recall': component_scores.get('dataset_recall', 0),
                'precision': component_scores.get('dataset_precision', 0),
                'f1': component_scores.get('dataset_f1', 0)
            }
        }

    return {
        'non_agentic': extract_component_alignment(non_agentic_align),
        'agentic': extract_component_alignment(agentic_align)
    }


def plot_quality_comparison_single(non_agentic_quality: float, agentic_quality: float) -> plt.Figure:
    """Create quality comparison bar chart for a single paper.

    Args:
        non_agentic_quality: Non-agentic quality score
        agentic_quality: Agentic quality score

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    quality_data = {
        'Non-Agentic': non_agentic_quality,
        'Agentic': agentic_quality
    }

    bars = ax.bar(quality_data.keys(), quality_data.values(),
                   color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
    ax.set_title('Setup Quality Scores', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good threshold (0.8)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_alignment_comparison_single(alignment_data: Dict) -> plt.Figure:
    """Create alignment comparison charts for a single paper.

    Args:
        alignment_data: Alignment data from get_alignment_data()

    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Overall alignment
    overall_scores = [
        alignment_data['non_agentic']['overall'],
        alignment_data['agentic']['overall']
    ]

    bars = ax1.bar(['Non-Agentic', 'Agentic'], overall_scores,
                   color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Alignment Score', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Alignment with Ground Truth', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Component breakdown
    components = ['baselines', 'metrics', 'datasets']
    component_labels = ['Baselines', 'Metrics', 'Datasets']
    x = range(len(components))
    width = 0.35

    non_agentic_scores = [
        alignment_data['non_agentic'][c].get('recall', 0)
        for c in components
    ]
    agentic_scores = [
        alignment_data['agentic'][c].get('recall', 0)
        for c in components
    ]

    ax2.bar([i - width/2 for i in x], non_agentic_scores, width,
            label='Non-Agentic', color='#3498db', edgecolor='black', linewidth=1)
    ax2.bar([i + width/2 for i in x], agentic_scores, width,
            label='Agentic', color='#e74c3c', edgecolor='black', linewidth=1)

    ax2.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_title('Component-wise Alignment', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(component_labels)
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_cost_comparison_single(non_agentic_meta: Dict, agentic_meta: Dict) -> plt.Figure:
    """Create cost comparison charts for a single paper.

    Args:
        non_agentic_meta: Non-agentic metadata
        agentic_meta: Agentic metadata

    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time comparison
    times = [
        non_agentic_meta.get('time_elapsed', 0),
        agentic_meta.get('time_elapsed', 0)
    ]

    bars1 = ax1.bar(['Non-Agentic', 'Agentic'], times,
                    color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Execution Time', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # LLM calls comparison
    calls = [
        non_agentic_meta.get('llm_calls', 0),
        agentic_meta.get('llm_calls', 0)
    ]

    bars2 = ax2.bar(['Non-Agentic', 'Agentic'], calls,
                    color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('LLM Calls', fontsize=12, fontweight='bold')
    ax2.set_title('Number of LLM Calls', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    return fig


def print_setup_components(components: Dict, title: str = "SETUP"):
    """Print formatted setup components.

    Args:
        components: Dictionary with baselines, metrics, datasets
        title: Title for the section
    """
    print("=" * 70)
    print(title)
    print("=" * 70)

    print("\nBASELINES:")
    print(format_component_list(components['baselines'], 'baseline'))

    print("\nMETRICS:")
    print(format_component_list(components['metrics'], 'metric'))

    print("\nDATASETS:")
    print(format_component_list(components['datasets'], 'dataset'))
    print()


def print_alignment_details(alignment_data: Dict, approach: str = "Non-Agentic"):
    """Print detailed alignment analysis.

    Args:
        alignment_data: Alignment data for one approach
        approach: Approach name (for display)
    """
    print(f"\n{'─' * 70}")
    print(f"{approach.upper()}")
    print(f"{'─' * 70}")

    print(f"\nOverall Alignment: {alignment_data['overall']:.1%}")

    for component_key, component_name in [
        ('baselines', 'BASELINES'),
        ('metrics', 'METRICS'),
        ('datasets', 'DATASETS')
    ]:
        comp_align = alignment_data.get(component_key, {})
        if not comp_align:
            continue

        print(f"\n{component_name}:")
        recall = comp_align.get('recall', 0)
        precision = comp_align.get('precision', 0)
        f1 = comp_align.get('f1', 0)

        print(f"  Recall:    {recall:.1%}")
        print(f"  Precision: {precision:.1%}")
        print(f"  F1 Score:  {f1:.1%}")


def generate_recommendation(
    non_agentic_quality: float,
    agentic_quality: float,
    alignment_data: Dict,
    cost_ratio_time: float,
    cost_ratio_calls: float
) -> str:
    """Generate recommendation for which approach is better.

    Args:
        non_agentic_quality: Non-agentic quality score
        agentic_quality: Agentic quality score
        alignment_data: Alignment data from get_alignment_data()
        cost_ratio_time: Time ratio (agentic/non-agentic)
        cost_ratio_calls: Calls ratio (agentic/non-agentic)

    Returns:
        Formatted recommendation string
    """
    non_align = alignment_data['non_agentic']['overall']
    agentic_align = alignment_data['agentic']['overall']

    # Determine winners
    quality_winner = 'agentic' if agentic_quality > non_agentic_quality else 'non_agentic'

    if non_align > 0 or agentic_align > 0:  # We have alignment data
        if agentic_align > non_align:
            alignment_winner = 'agentic'
        elif non_align > agentic_align:
            alignment_winner = 'non_agentic'
        else:
            alignment_winner = 'tie'
    else:
        alignment_winner = 'n/a'

    # Generate recommendation
    recommendation = []
    recommendation.append("=" * 70)
    recommendation.append("RECOMMENDATION")
    recommendation.append("=" * 70)

    if alignment_winner == 'agentic':
        recommendation.append("\n✅ RECOMMENDATION: Agentic approach is better for this paper")
        recommendation.append("  ✓ Better alignment with ground truth")
        recommendation.append(f"  • Note: {cost_ratio_time:.1f}x slower, {cost_ratio_calls:.1f}x more API calls")
    elif alignment_winner == 'non_agentic':
        recommendation.append("\n✅ RECOMMENDATION: Non-agentic approach is better for this paper")
        recommendation.append("  ✓ Better or equal alignment with ground truth")
        recommendation.append("  ✓ Much lower cost")
    elif quality_winner == 'non_agentic':
        recommendation.append("\n✅ RECOMMENDATION: Non-agentic approach is better for this paper")
        recommendation.append("  ✓ Better quality (no ground truth available for alignment)")
        recommendation.append("  ✓ Much lower cost")
    else:
        recommendation.append("\n⚖️  RECOMMENDATION: Agentic approach may be better for this paper")
        recommendation.append("  ✓ Better quality (no ground truth available for alignment)")
        recommendation.append(f"  • Note: {cost_ratio_time:.1f}x slower, {cost_ratio_calls:.1f}x more API calls")

    recommendation.append("=" * 70)

    return "\n".join(recommendation)


def load_single_paper_data(run_folder: str, paper_id: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Load data for a single paper from a run.

    Args:
        run_folder: Path to run folder
        paper_id: Paper ID to load

    Returns:
        Tuple of (results_dict, context_dict)
    """
    base_path = Path(run_folder)
    paper_path = base_path / paper_id

    if not paper_path.exists():
        print(f"❌ Paper folder not found: {paper_path}")
        return None, None

    results = load_json(paper_path / "results.json")
    context = load_json(paper_path / "paper_context.json")

    return results, context


def list_papers_in_run(run_folder: str) -> List[Dict]:
    """List all papers in a run with basic info.

    Args:
        run_folder: Path to run folder

    Returns:
        List of paper info dictionaries
    """
    base_path = Path(run_folder)

    if not base_path.exists():
        print(f"❌ Run folder not found: {base_path}")
        return []

    # Try to load from summary.json first (old format)
    summary = load_json(base_path / "summary.json")
    if summary and 'results' in summary:
        # Old format with results array
        papers_info = []
        for result in summary.get('results', []):
            if 'error' not in result:
                papers_info.append({
                    'id': result['paper_id'],
                    'title': result.get('title', 'Unknown')[:60],
                    'winner': result.get('winner', 'n/a'),
                    'non_agentic_quality': result.get('quality', {}).get('non_agentic', 0),
                    'agentic_quality': result.get('quality', {}).get('agentic', 0)
                })
        return papers_info

    # Fallback: scan directory for paper folders (new format)
    papers_info = []
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Try to load this paper's results.json
            results_file = item / "results.json"
            if results_file.exists():
                results = load_json(results_file)
                if results:
                    # Extract basic info from results
                    context_file = item / "paper_context.json"
                    context = load_json(context_file) if context_file.exists() else {}

                    non_agentic_quality = get_quality_score(results, APPROACH_NON_AGENTIC)
                    agentic_quality = get_quality_score(results, APPROACH_AGENTIC)

                    # Determine winner
                    winner = 'agentic' if agentic_quality > non_agentic_quality else 'non_agentic'

                    papers_info.append({
                        'id': item.name,
                        'title': context.get('title', 'Unknown')[:60],
                        'winner': winner,
                        'non_agentic_quality': non_agentic_quality,
                        'agentic_quality': agentic_quality
                    })

    return papers_info


def format_component_with_rationale(components: List, component_type: str = "item") -> str:
    """Format components with rationale for detailed text viewing.

    Args:
        components: List of components (can be strings or dicts)
        component_type: Type of component for display

    Returns:
        Formatted string with rationale
    """
    if not components:
        return f"No {component_type}s found"

    result = []
    for i, comp in enumerate(components, 1):
        if isinstance(comp, dict):
            name = comp.get('name', 'Unnamed')
            desc = comp.get('description', '')
            rationale = comp.get('rationale', '')
            complexity = comp.get('implementation_complexity', '')

            result.append(f"{i}. {name}")
            if desc:
                result.append(f"   Description: {desc}")
            if rationale:
                result.append(f"   Rationale: {rationale}")
            if complexity:
                result.append(f"   Complexity: {complexity}")
        else:
            result.append(f"{i}. {comp}")
        result.append("")  # Blank line

    return "\n".join(result)


def extract_component_names(components: List) -> List[str]:
    """Extract just the names from components.

    Args:
        components: List of components (can be strings or dicts)

    Returns:
        List of component names
    """
    names = []
    for comp in components:
        if isinstance(comp, dict):
            names.append(comp.get('name', 'Unnamed'))
        else:
            names.append(str(comp))
    return names


def create_side_by_side_comparison(
    non_agentic_components: Dict,
    agentic_components: Dict,
    ground_truth_components: Dict = None
) -> str:
    """Create three-way side-by-side comparison table.

    Args:
        non_agentic_components: Non-agentic components dict
        agentic_components: Agentic components dict
        ground_truth_components: Ground truth components dict (optional)

    Returns:
        Formatted comparison table
    """
    if ground_truth_components is None:
        ground_truth_components = {'baselines': [], 'metrics': [], 'datasets': []}

    # Extract names
    non_agentic_baselines = extract_component_names(non_agentic_components['baselines'])
    agentic_baselines = extract_component_names(agentic_components['baselines'])
    gt_baselines = extract_component_names(ground_truth_components['baselines'])

    non_agentic_metrics = extract_component_names(non_agentic_components['metrics'])
    agentic_metrics = extract_component_names(agentic_components['metrics'])
    gt_metrics = extract_component_names(ground_truth_components['metrics'])

    non_agentic_datasets = extract_component_names(non_agentic_components['datasets'])
    agentic_datasets = extract_component_names(agentic_components['datasets'])
    gt_datasets = extract_component_names(ground_truth_components['datasets'])

    lines = []
    lines.append("=" * 120)
    lines.append("THREE-WAY SIDE-BY-SIDE COMPARISON")
    lines.append("=" * 120)
    lines.append("")

    # Baselines
    lines.append("BASELINES:")
    lines.append("─" * 120)
    max_baselines = max(len(non_agentic_baselines), len(agentic_baselines), len(gt_baselines))
    lines.append(f"{'Non-Agentic':<35} | {'Agentic':<35} | {'Ground Truth (Actual)':<35}")
    lines.append("─" * 120)
    for i in range(max_baselines):
        na = non_agentic_baselines[i] if i < len(non_agentic_baselines) else ""
        ag = agentic_baselines[i] if i < len(agentic_baselines) else ""
        gt = gt_baselines[i] if i < len(gt_baselines) else ""
        # Truncate if too long
        na = na[:34] if len(na) > 34 else na
        ag = ag[:34] if len(ag) > 34 else ag
        gt = gt[:34] if len(gt) > 34 else gt
        lines.append(f"{na:<35} | {ag:<35} | {gt:<35}")
    lines.append("")

    # Metrics
    lines.append("METRICS:")
    lines.append("─" * 120)
    max_metrics = max(len(non_agentic_metrics), len(agentic_metrics), len(gt_metrics))
    lines.append(f"{'Non-Agentic':<35} | {'Agentic':<35} | {'Ground Truth (Actual)':<35}")
    lines.append("─" * 120)
    for i in range(max_metrics):
        na = non_agentic_metrics[i] if i < len(non_agentic_metrics) else ""
        ag = agentic_metrics[i] if i < len(agentic_metrics) else ""
        gt = gt_metrics[i] if i < len(gt_metrics) else ""
        na = na[:34] if len(na) > 34 else na
        ag = ag[:34] if len(ag) > 34 else ag
        gt = gt[:34] if len(gt) > 34 else gt
        lines.append(f"{na:<35} | {ag:<35} | {gt:<35}")
    lines.append("")

    # Datasets
    lines.append("DATASETS:")
    lines.append("─" * 120)
    max_datasets = max(len(non_agentic_datasets), len(agentic_datasets), len(gt_datasets))
    lines.append(f"{'Non-Agentic':<35} | {'Agentic':<35} | {'Ground Truth (Actual)':<35}")
    lines.append("─" * 120)
    for i in range(max_datasets):
        na = non_agentic_datasets[i] if i < len(non_agentic_datasets) else ""
        ag = agentic_datasets[i] if i < len(agentic_datasets) else ""
        gt = gt_datasets[i] if i < len(gt_datasets) else ""
        na = na[:34] if len(na) > 34 else na
        ag = ag[:34] if len(ag) > 34 else ag
        gt = gt[:34] if len(gt) > 34 else gt
        lines.append(f"{na:<35} | {ag:<35} | {gt:<35}")
    lines.append("")

    lines.append("=" * 120)
    lines.append("")

    # Summary
    lines.append("SUMMARY:")
    lines.append(f"  Non-Agentic: {len(non_agentic_baselines)} baselines, {len(non_agentic_metrics)} metrics, {len(non_agentic_datasets)} datasets")
    lines.append(f"  Agentic:     {len(agentic_baselines)} baselines, {len(agentic_metrics)} metrics, {len(agentic_datasets)} datasets")
    lines.append(f"  Ground Truth: {len(gt_baselines)} baselines, {len(gt_metrics)} metrics, {len(gt_datasets)} datasets")
    lines.append("=" * 120)

    return "\n".join(lines)


def compute_overlap_analysis(
    non_agentic_components: Dict,
    agentic_components: Dict
) -> str:
    """Compute overlap between non-agentic and agentic approaches.

    Args:
        non_agentic_components: Non-agentic components dict
        agentic_components: Agentic components dict

    Returns:
        Formatted overlap analysis
    """
    def normalize(name):
        """Normalize for comparison."""
        return name.lower().strip()

    def find_overlap(list1, list2):
        """Find shared and unique items."""
        set1 = {normalize(x) for x in list1}
        set2 = {normalize(x) for x in list2}

        shared = set1 & set2
        only_1 = set1 - set2
        only_2 = set2 - set1

        return sorted(shared), sorted(only_1), sorted(only_2)

    # Extract names
    non_agentic_baselines = extract_component_names(non_agentic_components['baselines'])
    agentic_baselines = extract_component_names(agentic_components['baselines'])
    non_agentic_metrics = extract_component_names(non_agentic_components['metrics'])
    agentic_metrics = extract_component_names(agentic_components['metrics'])
    non_agentic_datasets = extract_component_names(non_agentic_components['datasets'])
    agentic_datasets = extract_component_names(agentic_components['datasets'])

    lines = []
    lines.append("=" * 80)
    lines.append("OVERLAP ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    # Baselines
    shared_b, only_na_b, only_ag_b = find_overlap(non_agentic_baselines, agentic_baselines)
    lines.append("BASELINES:")
    lines.append(f"  Shared: {len(shared_b)}")
    if shared_b:
        for item in shared_b:
            lines.append(f"    - {item}")
    lines.append(f"  Only Non-Agentic: {len(only_na_b)}")
    if only_na_b:
        for item in only_na_b:
            lines.append(f"    - {item}")
    lines.append(f"  Only Agentic: {len(only_ag_b)}")
    if only_ag_b:
        for item in only_ag_b:
            lines.append(f"    - {item}")
    lines.append("")

    # Metrics
    shared_m, only_na_m, only_ag_m = find_overlap(non_agentic_metrics, agentic_metrics)
    lines.append("METRICS:")
    lines.append(f"  Shared: {len(shared_m)}")
    if shared_m:
        for item in shared_m:
            lines.append(f"    - {item}")
    lines.append(f"  Only Non-Agentic: {len(only_na_m)}")
    if only_na_m:
        for item in only_na_m:
            lines.append(f"    - {item}")
    lines.append(f"  Only Agentic: {len(only_ag_m)}")
    if only_ag_m:
        for item in only_ag_m:
            lines.append(f"    - {item}")
    lines.append("")

    # Datasets
    shared_d, only_na_d, only_ag_d = find_overlap(non_agentic_datasets, agentic_datasets)
    lines.append("DATASETS:")
    lines.append(f"  Shared: {len(shared_d)}")
    if shared_d:
        for item in shared_d:
            lines.append(f"    - {item}")
    lines.append(f"  Only Non-Agentic: {len(only_na_d)}")
    if only_na_d:
        for item in only_na_d:
            lines.append(f"    - {item}")
    lines.append(f"  Only Agentic: {len(only_ag_d)}")
    if only_ag_d:
        for item in only_ag_d:
            lines.append(f"    - {item}")

    lines.append("=" * 80)

    return "\n".join(lines)
