#!/usr/bin/env python3
"""
Paper Extraction Validator and Filter

Validates that papers have been successfully extracted with all required sections
and generates a filtered papers_to_process.txt with only high-quality papers.

This validator uses the ACTUAL PaperProcessorTool to ensure validation matches
the real extraction logic used during pipeline execution.

Usage:
    # Validate a specific collection (no LLM calls - just parsing check)
    python experimental_setup_eval/validate_and_filter_papers.py \
        data/papers/20251119_161731_collection

    # Validate with custom thresholds
    python experimental_setup_eval/validate_and_filter_papers.py \
        data/papers/20251119_161731_collection \
        --min-title-length 15 \
        --min-abstract-length 200 \
        --min-intro-length 500

Output:
    - Creates validated_papers_to_process.txt with only papers passing validation
    - Creates validation_report.json with detailed results for each paper
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Import actual paper processor
sys.path.insert(0, str(Path(__file__).parent.parent))
from experimental_setup_eval.tools.paper_processor import PaperProcessorTool


@dataclass
class ValidationResult:
    """Results of paper extraction validation."""
    paper_path: str
    paper_id: str
    passed: bool
    score: float  # 0-100
    checks: Dict[str, bool]
    issues: List[str]
    warnings: List[str]
    metadata: Dict[str, any]


class PaperExtractionValidator:
    """Validates quality of extracted paper content using actual PaperProcessorTool."""

    def __init__(
        self,
        min_title_length: int = 10,
        min_abstract_length: int = 100,
        min_intro_length: int = 300,
        min_method_length: int = 200,
        min_total_content: int = 1000,
    ):
        """
        Initialize validator with quality thresholds.

        Args:
            min_title_length: Minimum characters for title
            min_abstract_length: Minimum characters for abstract
            min_intro_length: Minimum characters for introduction
            min_method_length: Minimum characters for method description
            min_total_content: Minimum total characters across all sections
        """
        self.min_title_length = min_title_length
        self.min_abstract_length = min_abstract_length
        self.min_intro_length = min_intro_length
        self.min_method_length = min_method_length
        self.min_total_content = min_total_content

        # Create processor instance without model (we only need parsing, not LLM calls)
        self.processor = PaperProcessorTool.__new__(PaperProcessorTool)

    def validate_paper(self, paper_md_path: Path) -> ValidationResult:
        """
        Validate a single paper's extraction quality.

        Uses actual PaperProcessorTool parsing logic to match pipeline behavior.

        Args:
            paper_md_path: Path to paper.md file

        Returns:
            ValidationResult with pass/fail and detailed feedback
        """
        paper_dir = paper_md_path.parent
        paper_id = paper_dir.name

        checks = {}
        issues = []
        warnings = []
        metadata = {}

        # Check 1: Paper markdown exists
        if not paper_md_path.exists():
            return ValidationResult(
                paper_path=str(paper_md_path),
                paper_id=paper_id,
                passed=False,
                score=0.0,
                checks={"markdown_exists": False},
                issues=["Paper markdown file not found"],
                warnings=[],
                metadata={}
            )
        checks["markdown_exists"] = True

        # Check 2: Read and parse markdown
        try:
            with open(paper_md_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            metadata["markdown_size"] = len(markdown_content)
            checks["readable"] = True
        except Exception as e:
            issues.append(f"Cannot read markdown: {e}")
            checks["readable"] = False
            return ValidationResult(
                paper_path=str(paper_md_path),
                paper_id=paper_id,
                passed=False,
                score=10.0,
                checks=checks,
                issues=issues,
                warnings=warnings,
                metadata=metadata
            )

        # Check 3: Minimum file size (catch empty or corrupted files)
        min_file_size = 1000  # bytes
        if len(markdown_content) < min_file_size:
            issues.append(f"Markdown too small ({len(markdown_content)} < {min_file_size} bytes)")
            checks["min_size"] = False
        else:
            checks["min_size"] = True

        # Check 4: Load metadata if available
        metadata_path = paper_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    paper_metadata = json.load(f)
                metadata["has_metadata"] = True
                metadata["source_title"] = paper_metadata.get("title", "")
                metadata["year"] = paper_metadata.get("year", 0)
                metadata["citations"] = paper_metadata.get("citationCount", 0)
            except:
                warnings.append("Could not parse metadata.json")
                metadata["has_metadata"] = False
        else:
            warnings.append("No metadata.json found")
            metadata["has_metadata"] = False

        # Check 5: Use ACTUAL PaperProcessorTool parsing
        try:
            # Clean markdown (using actual processor method)
            cleaned_content = self.processor._clean_markdown(markdown_content)
            metadata["cleaned_size"] = len(cleaned_content)
            metadata["removed_html"] = len(markdown_content) - len(cleaned_content)

            # Parse sections (using actual processor method)
            sections = self.processor._parse_sections(markdown_content)
            metadata["sections_found"] = list(sections.keys())
            metadata["section_lengths"] = {k: len(v) for k, v in sections.items()}

            # Extract key sections (using actual processor method)
            title = self.processor._extract_title(sections)

            # Extract sections using processor's helper
            def find_section(keyword: str) -> str:
                for section_name, content in sections.items():
                    if keyword in section_name.lower():
                        return content
                return ""

            abstract = find_section("abstract")
            introduction = find_section("introduction")
            related_work = find_section("related")
            method = (
                find_section("method") or
                find_section("approach") or
                find_section("model")
            )

            metadata["extracted_title"] = title[:100] if title else ""
            metadata["extracted_lengths"] = {
                "title": len(title),
                "abstract": len(abstract),
                "introduction": len(introduction),
                "related_work": len(related_work),
                "method": len(method),
            }

            checks["parsing_successful"] = True

        except Exception as e:
            issues.append(f"Parsing failed: {e}")
            checks["parsing_successful"] = False
            return ValidationResult(
                paper_path=str(paper_md_path),
                paper_id=paper_id,
                passed=False,
                score=20.0,
                checks=checks,
                issues=issues,
                warnings=warnings,
                metadata=metadata
            )

        # Check 6: Title quality
        if not title or len(title) < self.min_title_length:
            issues.append(f"Title too short or missing ({len(title)} chars)")
            checks["title_ok"] = False
        elif title in ["Unknown Title", "_2024-10-23_", "header"]:
            issues.append(f"Invalid title detected: '{title}'")
            checks["title_ok"] = False
        else:
            checks["title_ok"] = True

        # Check 7: Abstract quality
        if len(abstract) < self.min_abstract_length:
            issues.append(f"Abstract too short ({len(abstract)} < {self.min_abstract_length} chars)")
            checks["abstract_ok"] = False
        else:
            checks["abstract_ok"] = True

        # Check 8: Introduction quality
        if len(introduction) < self.min_intro_length:
            issues.append(f"Introduction too short ({len(introduction)} < {self.min_intro_length} chars)")
            checks["intro_ok"] = False
        else:
            checks["intro_ok"] = True

        # Check 9: Related work (warn if missing, don't fail)
        if len(related_work) < 100:
            warnings.append("Related work section missing (may be embedded in introduction)")
            checks["related_work_ok"] = False
        else:
            checks["related_work_ok"] = True

        # Check 10: Method description
        if len(method) < self.min_method_length:
            issues.append(f"Method section too short ({len(method)} < {self.min_method_length} chars)")
            checks["method_ok"] = False
        else:
            checks["method_ok"] = True

        # Check 11: Total content sufficiency
        total_content = len(title) + len(abstract) + len(introduction) + len(related_work) + len(method)
        metadata["total_content_length"] = total_content

        if total_content < self.min_total_content:
            issues.append(f"Total content too short ({total_content} < {self.min_total_content} chars)")
            checks["content_sufficient"] = False
        else:
            checks["content_sufficient"] = True

        # Check 12: Compare extracted title with metadata title (if available)
        if metadata.get("has_metadata") and metadata.get("source_title"):
            source_title_lower = metadata["source_title"].lower()
            extracted_title_lower = title.lower()

            # Simple similarity check
            title_words_source = set(source_title_lower.split())
            title_words_extracted = set(extracted_title_lower.split())

            if title_words_source and title_words_extracted:
                overlap = len(title_words_source & title_words_extracted) / len(title_words_source)
                metadata["title_overlap"] = overlap

                if overlap < 0.3:  # Less than 30% word overlap
                    warnings.append(f"Extracted title differs from metadata (overlap: {overlap:.1%})")

        # Calculate overall score
        total_checks = len([v for v in checks.values() if isinstance(v, bool)])
        passed_checks = sum([1 for v in checks.values() if v is True])
        score = (passed_checks / total_checks * 100) if total_checks > 0 else 0

        # Determine pass/fail
        # Required checks for passing:
        critical_checks = ["markdown_exists", "readable", "min_size", "parsing_successful",
                          "title_ok", "abstract_ok", "intro_ok", "method_ok", "content_sufficient"]
        passed = all(checks.get(check, False) for check in critical_checks)

        return ValidationResult(
            paper_path=str(paper_md_path),
            paper_id=paper_id,
            passed=passed,
            score=score,
            checks=checks,
            issues=issues,
            warnings=warnings,
            metadata=metadata
        )


def validate_collection(
    collection_dir: Path,
    validator: PaperExtractionValidator,
    output_dir: Optional[Path] = None
) -> Tuple[List[ValidationResult], Dict]:
    """
    Validate all papers in a collection directory.

    Args:
        collection_dir: Path to collection directory
        validator: Validator instance
        output_dir: Optional output directory (defaults to collection_dir)

    Returns:
        Tuple of (validation results list, summary dict)
    """
    if output_dir is None:
        output_dir = collection_dir

    # Find all paper.md files
    paper_paths = list(collection_dir.glob("*/paper.md"))

    print(f"Found {len(paper_paths)} papers in {collection_dir.name}")
    print()

    results = []
    for i, paper_path in enumerate(paper_paths, 1):
        paper_id = paper_path.parent.name
        print(f"[{i}/{len(paper_paths)}] Validating {paper_id[:60]}...", end=" ")

        result = validator.validate_paper(paper_path)
        results.append(result)

        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{status} ({result.score:.0f}%)")

        if result.issues:
            for issue in result.issues[:3]:  # Show first 3 issues
                print(f"    ⚠ {issue}")
        if result.warnings and len(result.warnings) <= 2:
            for warning in result.warnings:
                print(f"    ℹ {warning}")

    # Generate summary
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    summary = {
        "total_papers": len(results),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": len(passed) / len(results) * 100 if results else 0,
        "average_score": sum(r.score for r in results) / len(results) if results else 0,
        "common_issues": _get_common_issues(failed),
    }

    return results, summary


def _get_common_issues(failed_results: List[ValidationResult]) -> List[Dict]:
    """Extract most common issues from failed validations."""
    from collections import Counter

    all_issues = []
    for result in failed_results:
        all_issues.extend(result.issues)

    issue_counts = Counter(all_issues)
    return [
        {"issue": issue, "count": count}
        for issue, count in issue_counts.most_common(10)
    ]


def write_filtered_papers(
    results: List[ValidationResult],
    collection_dir: Path,
    output_path: Path,
    query: str = "unknown"
):
    """
    Write filtered papers_to_process.txt with only validated papers.

    Args:
        results: Validation results
        collection_dir: Collection directory
        output_path: Output file path
        query: Search query for header
    """
    from datetime import datetime

    passed_results = [r for r in results if r.passed]
    failed_results = [r for r in results if not r.passed]

    with open(output_path, 'w') as f:
        f.write("# Validated papers for processing\n")
        f.write(f"# Collection: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Query: {query}\n")
        f.write(f"# Validated: {len(passed_results)}/{len(results)} papers passed\n")
        f.write("#\n")
        f.write("# Use with: python experimental_setup_eval/run_pipeline.py validated_papers_to_process.txt\n")
        f.write("#\n\n")

        # Write passed papers
        for result in passed_results:
            f.write(f"{result.paper_path}\n")

        # Write failed papers as comments
        if failed_results:
            f.write("\n# Failed validation:\n")
            for result in failed_results:
                # Write primary issue
                primary_issue = result.issues[0] if result.issues else "Unknown issue"
                f.write(f"# FAILED ({result.score:.0f}%): {result.paper_id[:50]} - {primary_issue}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate paper extractions and filter for processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "collection_dir",
        type=Path,
        help="Path to paper collection directory"
    )
    parser.add_argument(
        "--min-title-length",
        type=int,
        default=10,
        help="Minimum title length (default: 10)"
    )
    parser.add_argument(
        "--min-abstract-length",
        type=int,
        default=100,
        help="Minimum abstract length (default: 100)"
    )
    parser.add_argument(
        "--min-intro-length",
        type=int,
        default=300,
        help="Minimum introduction length (default: 300)"
    )
    parser.add_argument(
        "--min-method-length",
        type=int,
        default=200,
        help="Minimum method section length (default: 200)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: collection_dir)"
    )

    args = parser.parse_args()

    if not args.collection_dir.exists():
        print(f"Error: Collection directory not found: {args.collection_dir}")
        sys.exit(1)

    # Initialize validator
    validator = PaperExtractionValidator(
        min_title_length=args.min_title_length,
        min_abstract_length=args.min_abstract_length,
        min_intro_length=args.min_intro_length,
        min_method_length=args.min_method_length,
    )

    # Run validation
    print("=" * 70)
    print("PAPER EXTRACTION VALIDATION")
    print("=" * 70)
    print()

    results, summary = validate_collection(
        args.collection_dir,
        validator,
        args.output_dir
    )

    # Write outputs
    output_dir = args.output_dir or args.collection_dir

    # Write detailed report
    report_path = output_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            "summary": summary,
            "results": [asdict(r) for r in results]
        }, f, indent=2)

    # Write filtered papers list
    filtered_path = output_dir / "validated_papers_to_process.txt"

    # Try to get query from original papers_to_process.txt
    original_papers_file = args.collection_dir / "papers_to_process.txt"
    query = "unknown"
    if original_papers_file.exists():
        with open(original_papers_file, 'r') as f:
            for line in f:
                if line.startswith("# Query:"):
                    query = line.replace("# Query:", "").strip()
                    break

    write_filtered_papers(results, args.collection_dir, filtered_path, query)

    # Print summary
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total papers:    {summary['total_papers']}")
    print(f"Passed:          {summary['passed']} ({summary['pass_rate']:.1f}%)")
    print(f"Failed:          {summary['failed']}")
    print(f"Average score:   {summary['average_score']:.1f}%")
    print()

    if summary['common_issues']:
        print("Most common issues:")
        for issue_info in summary['common_issues'][:5]:
            print(f"  • {issue_info['issue']} ({issue_info['count']}x)")
        print()

    print(f"✓ Validation report: {report_path}")
    print(f"✓ Filtered papers:   {filtered_path}")
    print()

    # Print ranked paper list
    print("=" * 70)
    print("PAPER RANKING BY EXTRACTION QUALITY")
    print("=" * 70)
    print()

    # Sort results by score (descending)
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

    for i, result in enumerate(sorted_results, 1):
        status_icon = "✓" if result.passed else "✗"
        status_text = "PASS" if result.passed else "FAIL"

        # Get source metadata if available
        source_title = result.metadata.get("source_title", "")
        year = result.metadata.get("year", "")
        citations = result.metadata.get("citations", 0)

        # Format metadata
        meta_info = []
        if year:
            meta_info.append(f"{year}")
        if citations:
            meta_info.append(f"{citations} citations")
        meta_str = ", ".join(meta_info) if meta_info else ""

        print(f"#{i}. {status_icon} {status_text} ({result.score:.0f}%) - {result.paper_id[:55]}")
        if source_title:
            print(f"    Title: {source_title[:70]}")
        if meta_str:
            print(f"    Meta:  {meta_str}")

        # Show extraction stats
        lengths = result.metadata.get("extracted_lengths", {})
        if lengths:
            print(f"    Extracted: Abstract={lengths.get('abstract', 0)}ch, "
                  f"Intro={lengths.get('introduction', 0)}ch, "
                  f"Method={lengths.get('method', 0)}ch")

        # Show top issue if failed
        if not result.passed and result.issues:
            print(f"    Issue: {result.issues[0]}")

        print()

    print("=" * 70)
    print()

    if summary['passed'] > 0:
        print(f"✓ {summary['passed']} papers ready for processing!")
        print(f"\nNext: python experimental_setup_eval/run_pipeline.py {filtered_path}")
    else:
        print("⚠️  No papers passed validation. Check common issues above.")
        print("\nConsider:")
        print("  • Lowering thresholds with --min-*-length flags")
        print("  • Checking PDF-to-markdown conversion quality")
        print("  • Reviewing validation_report.json for details")

    print("=" * 70)


if __name__ == "__main__":
    main()
