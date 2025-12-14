"""
Simplified Experimental Setup Evaluation Pipeline v2

Clean, maintainable pipeline using unified architecture:
- OmegaConf configuration with CLI support
- Unified evaluator replacing 4 separate classes
- Clean orchestrator replacing complex main() function
- Typed data models for better code clarity

Usage (run from experimental_setup_eval/ directory):
    cd experimental_setup_eval/
    python run_pipeline.py ../data/papers/collection/papers.txt
    python run_pipeline.py ../data/papers/collection/papers.txt model=gpt-4o
    python run_pipeline.py ../data/papers/collection/papers.txt debug=true
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ai-scientist")

# Add parent directory to path for ai_scientist imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Direct imports avoiding circular dependencies
# Direct imports avoiding circular dependencies
from experimental_setup_eval.core.config import load_config, get_default_config
from experimental_setup_eval.core.pipeline import SimpleEvaluationPipeline, generate_summary, print_final_summary


def main():
    """Clean main function using new unified architecture."""
    if len(sys.argv) < 2 or sys.argv[1] in ['--help', '-h']:
        print("Usage: python run_pipeline.py <papers_file> [config_overrides...]")
        print("Examples:")
        print("  python run_pipeline.py papers.txt")
        print("  python run_pipeline.py papers.txt model=gpt-4o")
        print("  python run_pipeline.py papers.txt debug=true evaluation.agentic.num_drafts=5")
        print("\nConfiguration file: config.yaml")
        print("Available models: llama-3.1-8b-instant (default), gpt-4o, claude-sonnet, etc.")
        sys.exit(1)

    papers_file = sys.argv[1]
    config_overrides = sys.argv[2:] if len(sys.argv) > 2 else []

    # Load configuration with CLI argument support
    try:
        # Only use CLI args if there are actual config overrides
        use_cli_args = len(config_overrides) > 0
        if use_cli_args:
            # Set sys.argv to only include config overrides for OmegaConf
            original_argv = sys.argv
            sys.argv = [sys.argv[0]] + config_overrides
            try:
                config = load_config(use_cli_args=True)
            finally:
                sys.argv = original_argv
        else:
            config = load_config(use_cli_args=False)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("\nIf this is a CLI argument parsing error, try using config file overrides instead:")
        print("  python run_pipeline.py papers.txt model=gpt-4o")
        sys.exit(1)

    # Load papers
    papers = load_papers(papers_file)

    # Get model from config (with API key validation)
    try:
        from experimental_setup_eval.core.config import get_default_config
        config_loader = get_default_config()
        
        # Use the model from the loaded config (which includes CLI overrides)
        requested_model = config.models.default
        
        if config_loader.check_model_availability(requested_model):
            model = requested_model
        else:
            # Strict enforcement: No fallback
            print(f"Error: Requested model '{requested_model}' not available (missing API key).")
            print(f"Please set the appropriate API key for {requested_model} (e.g. export ANTHROPIC_API_KEY=...)")
            sys.exit(1)
            
        print(f"✓ Using model: {model}")
    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nPlease set at least one API key:")
        print("  For Groq (recommended - fast & cost-effective):")
        print("    export GROQ_API_KEY='your-groq-key'")
        print("  For OpenAI:")
        print("    export OPENAI_API_KEY='your-openai-key'")
        print("  For Anthropic:")
        print("    export ANTHROPIC_API_KEY='your-anthropic-key'")
        sys.exit(1)

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Resolve output path relative to script directory
    base_dir = Path(__file__).parent.resolve()
    output_pattern = config.pipeline.output_pattern
    
    # Handle output pattern
    if "{timestamp}" in output_pattern:
        output_path_str = output_pattern.format(timestamp=timestamp)
    else:
        output_path_str = output_pattern
        
    run_dir = base_dir / output_path_str
    run_dir.mkdir(parents=True, exist_ok=True)

    import os
    try:
        rel_output = os.path.relpath(run_dir)
    except ValueError:
        rel_output = str(run_dir)
        
    print(f"\n{'='*70}")
    print(f"UNIFIED EVALUATION PIPELINE: {len(papers)} papers")
    print(f"Output: {rel_output}")
    print(f"{'='*70}\n")

    # Create pipeline with configuration
    pipeline = SimpleEvaluationPipeline(config, model)

    # Process papers
    results = []
    for i, paper_path in enumerate(papers, 1):
        paper_id = paper_path.parent.name

        print(f"\n{'='*70}")
        print(f"[{i}/{len(papers)}] {paper_id[:50]}")
        print(f"{'='*70}")

        # Create paper-specific output directory
        output_dir = run_dir / paper_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process paper
        result = pipeline.run_single_paper(paper_path, output_dir)
        results.append(result)

        # Print summary
        if result["success"]:
            print(f"\n✓ Success ({result['processing_time']:.1f}s)")
            print(f"  Quality: Non-agentic {result['quality']['non_agentic']:.3f}, Agentic {result['quality']['agentic']:.3f}")
            if result["has_experiments"]:
                print(f"  Alignment winner: {result['alignment']['winner'].upper()}")
        else:
            print(f"\n✗ Failed: {result['error']}")

    # Generate summary
    summary = generate_summary(results, timestamp, str(run_dir))

    # Save summary
    with open(run_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Print final results
    # Use relative path for summary if possible
    import os
    try:
        rel_output = os.path.relpath(run_dir)
        summary['run_dir'] = rel_output # Update summary with relative path for consistency
    except ValueError:
        rel_output = str(run_dir)
    
    print_final_summary(summary, results)


def load_papers(papers_file: str) -> List[Path]:
    """Load and validate paper paths."""
    base_dir = Path(__file__).parent.resolve()
    
    # Resolve papers_file path
    papers_file_path = Path(papers_file)
    if not papers_file_path.is_absolute():
        papers_file_path = base_dir / papers_file
        
    try:
        with open(papers_file_path) as f:
            paper_paths = []
            for line in f:
                if not line.strip() or line.startswith('#'):
                    continue
                    
                path = Path(line.strip())
                # If path is relative, try resolving it relative to base_dir
                if not path.is_absolute():
                    resolved_path = base_dir / path
                    if resolved_path.exists():
                        paper_paths.append(resolved_path)
                    else:
                        # Fallback: try resolving relative to papers_file location
                        resolved_path_2 = papers_file_path.parent / path
                        if resolved_path_2.exists():
                            paper_paths.append(resolved_path_2)
                        else:
                            # Keep original for error reporting
                            paper_paths.append(path)
                else:
                    paper_paths.append(path)

        # Validate paths exist
        missing = [p for p in paper_paths if not p.exists()]
        if missing:
            print(f"Warning: {len(missing)} paper files not found")
            if len(missing) <= 3:
                for p in missing:
                    print(f"  Missing: {p}")

        valid_papers = [p for p in paper_paths if p.exists()]
        print(f"Loaded {len(valid_papers)} valid papers from {papers_file}")

        return valid_papers

    except FileNotFoundError:
        print(f"Error: Papers file '{papers_file}' not found")
        sys.exit(1)





if __name__ == "__main__":
    main()