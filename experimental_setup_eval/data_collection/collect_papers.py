#!/usr/bin/env python3
"""
Paper collection tool using Semantic Scholar API.

Collects research papers with PDF downloads and markdown conversion.
Each run creates a timestamped folder for organized output.

Usage:
    # Basic search with 2 papers
    python experimental_setup_eval/data_collection/collect_papers.py \
        --query "transformer neural networks" \
        --max-results 2 \
        --download-pdfs \
        --convert-markdown

    # Field-based search
    python experimental_setup_eval/data_collection/collect_papers.py \
        --query "attention mechanism computer vision" \
        --field "Computer Science" \
        --year-min 2017 \
        --min-citations 100 \
        --max-results 2 \
        --download-pdfs \
        --convert-markdown

    # Quick trial with 2 papers
    python experimental_setup_eval/data_collection/collect_papers.py \
        --query "BERT language model" \
        --max-results 2 \
        --download-pdfs \
        --convert-markdown \
        --output-dir data/trial_papers

Examples for AI/ML papers:
    # Computer Vision
    python experimental_setup_eval/data_collection/collect_papers.py \
        --query "ResNet image classification" \
        --max-results 2 --download-pdfs --convert-markdown

    # NLP
    python experimental_setup_eval/data_collection/collect_papers.py \
        --query "transformer machine translation" \
        --max-results 2 --download-pdfs --convert-markdown

    # Reinforcement Learning
    python experimental_setup_eval/data_collection/collect_papers.py \
        --query "deep reinforcement learning Atari" \
        --max-results 2 --download-pdfs --convert-markdown
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Try to import pymupdf4llm for markdown conversion
try:
    import pymupdf4llm
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: pymupdf4llm not installed. Markdown conversion disabled.")
    print("Install with: pip install pymupdf4llm")


class PaperCollector:
    """Collect papers using Semantic Scholar API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        throttle_seconds: float = 2.0,
        base_output_dir: str = "data/papers"
    ):
        """
        Initialize paper collector.

        Args:
            api_key: Semantic Scholar API key (or use S2_API_KEY env var)
            throttle_seconds: Minimum seconds between requests (default: 2.0)
            base_output_dir: Base directory for all outputs (default: data/papers)
        """
        self.api_key = api_key or os.getenv("S2_API_KEY")
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {}
        self.throttle_seconds = throttle_seconds
        self.last_request_time = 0

        # Create timestamped folder for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_output_dir) / f"{timestamp}_collection"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        print(f"✓ Output directory: {self.run_dir}/")

        if self.api_key:
            self.headers["x-api-key"] = self.api_key
            print(f"✓ Using Semantic Scholar API key")
            print(f"✓ Rate limiting: 1 request per {throttle_seconds} seconds")
        else:
            print("⚠ WARNING: No API key found. Rate limits will be stricter.")
            print(f"  Set S2_API_KEY environment variable for higher rate limits")
            print(f"  Apply at: https://www.semanticscholar.org/product/api")
            print(f"✓ Rate limiting: 1 request per {throttle_seconds} seconds")

    def _throttle(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.throttle_seconds:
            sleep_time = self.throttle_seconds - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def search_papers(
        self,
        query: str,
        max_results: int = 100,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        min_citations: Optional[int] = None,
        fields_of_study: Optional[List[str]] = None,
        venue: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for papers using Semantic Scholar API.

        Args:
            query: Search query string
            max_results: Maximum number of results to retrieve
            year_min: Minimum publication year
            year_max: Maximum publication year
            min_citations: Minimum citation count
            fields_of_study: List of fields (e.g., ["Computer Science"])
            venue: Filter by venue (e.g., "NeurIPS", "ICML")

        Returns:
            List of paper dictionaries
        """
        fields = [
            "paperId", "title", "abstract", "year", "authors",
            "citationCount", "venue", "publicationTypes",
            "publicationDate", "externalIds", "url", "openAccessPdf",
            "fieldsOfStudy", "s2FieldsOfStudy"
        ]

        all_papers = []
        offset = 0
        batch_size = 100  # API max per request

        print(f"\n{'='*70}")
        print(f"SEARCHING PAPERS")
        print(f"{'='*70}")
        print(f"Query: '{query}'")
        print(f"Max results: {max_results}")
        if year_min or year_max:
            print(f"Year range: {year_min or 'any'} - {year_max or 'any'}")
        if min_citations:
            print(f"Min citations: {min_citations}")
        if fields_of_study:
            print(f"Fields: {', '.join(fields_of_study)}")
        print()

        while len(all_papers) < max_results:
            params = {
                "query": query,
                "limit": min(batch_size, max_results - len(all_papers)),
                "offset": offset,
                "fields": ",".join(fields)
            }

            # Add filters
            filters = []
            if year_min or year_max:
                year_filter = f"{year_min or 1900}-{year_max or 2025}"
                params["year"] = year_filter

            if venue:
                params["venue"] = venue

            try:
                # Enforce rate limiting
                self._throttle()

                response = requests.get(
                    f"{self.base_url}/paper/search",
                    headers=self.headers,
                    params=params,
                    timeout=30
                )

                if response.status_code == 429:
                    print("⚠ Rate limit hit. Waiting 10 seconds...")
                    time.sleep(10)
                    continue

                response.raise_for_status()
                data = response.json()

                papers = data.get("data", [])
                if not papers:
                    break

                # Apply post-filters
                for paper in papers:
                    # Filter by citations
                    if min_citations and paper.get("citationCount", 0) < min_citations:
                        continue

                    # Filter by fields of study
                    if fields_of_study:
                        paper_fields = [f.lower() for f in paper.get("fieldsOfStudy", [])]
                        if not any(f.lower() in paper_fields for f in fields_of_study):
                            continue

                    all_papers.append(paper)

                print(f"Retrieved {len(all_papers)} papers so far...")

                offset += len(papers)

                # Check if we've reached the end
                if len(papers) < batch_size:
                    break

            except Exception as e:
                print(f"Error during search: {e}")
                break

        print(f"\nTotal papers found: {len(all_papers)}")
        return all_papers[:max_results]

    def download_pdf(
        self,
        paper: Dict,
        output_folder: Path
    ) -> Optional[Path]:
        """
        Download PDF for a paper.

        Args:
            paper: Paper dictionary from Semantic Scholar
            output_folder: Folder to save PDF

        Returns:
            Path to downloaded PDF, or None if unavailable
        """
        # Try openAccessPdf first
        open_access = paper.get("openAccessPdf")
        pdf_url = open_access.get("url") if open_access else None

        # Fallback to ArXiv
        if not pdf_url:
            external_ids = paper.get("externalIds", {})
            arxiv_id = external_ids.get("ArXiv")
            if arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                print(f"  → Using ArXiv URL")

        if not pdf_url:
            print(f"  ⊘ No PDF URL available")
            return None

        pdf_path = output_folder / "paper.pdf"

        try:
            self._throttle()

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/pdf'
            }

            response = requests.get(pdf_url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()

            content = response.content

            # Verify it's actually a PDF
            if not content.startswith(b'%PDF'):
                print(f"  ✗ Not a valid PDF file")
                return None

            # Save PDF
            with open(pdf_path, 'wb') as f:
                f.write(content)

            print(f"  ✓ PDF saved: {pdf_path.name} ({len(content):,} bytes)")
            return pdf_path

        except Exception as e:
            print(f"  ✗ PDF download failed: {e}")
            return None

    def convert_to_markdown(
        self,
        pdf_path: Path,
        output_folder: Path
    ) -> Optional[Path]:
        """
        Convert PDF to markdown using pymupdf4llm.

        Args:
            pdf_path: Path to PDF file
            output_folder: Folder to save markdown

        Returns:
            Path to markdown file, or None if conversion failed
        """
        if not HAS_PYMUPDF:
            print(f"  ⊘ pymupdf4llm not installed, skipping markdown conversion")
            return None

        markdown_path = output_folder / "paper.md"

        try:
            markdown_text = pymupdf4llm.to_markdown(str(pdf_path))

            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)

            print(f"  ✓ Markdown saved: {markdown_path.name}")
            return markdown_path

        except Exception as e:
            print(f"  ✗ Markdown conversion failed: {e}")
            return None

    def download_arxiv_source(
        self,
        arxiv_id: str,
        output_folder: Path
    ) -> Optional[Path]:
        """
        Download ArXiv LaTeX source files.

        Args:
            arxiv_id: ArXiv paper ID
            output_folder: Folder to save source

        Returns:
            Path to latex_source directory, or None if unavailable
        """
        # Clean ArXiv ID (remove version if present)
        clean_id = arxiv_id.split('v')[0]
        source_url = f"https://arxiv.org/e-print/{clean_id}"

        latex_dir = output_folder / "latex_source"
        latex_dir.mkdir(exist_ok=True)

        try:
            self._throttle()

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(source_url, headers=headers, timeout=30)
            response.raise_for_status()

            # Try to extract tar.gz
            try:
                import tarfile
                import io

                with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:gz') as tar:
                    tar.extractall(path=latex_dir)

                print(f"  ✓ LaTeX source extracted to: latex_source/")
                return latex_dir

            except Exception as e:
                # Maybe it's just a .tex file
                tex_path = latex_dir / "main.tex"
                with open(tex_path, 'wb') as f:
                    f.write(response.content)
                print(f"  ✓ LaTeX source saved: latex_source/main.tex")
                return latex_dir

        except Exception as e:
            print(f"  ⊘ ArXiv source download failed: {e}")
            return None

    def collect_papers(
        self,
        query: str,
        max_results: int = 2,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        min_citations: Optional[int] = None,
        fields_of_study: Optional[List[str]] = None,
        venue: Optional[str] = None,
        download_pdfs: bool = True,
        convert_markdown: bool = True,
        download_latex: bool = False
    ) -> List[Dict]:
        """
        Complete collection workflow: search + download + convert.

        Args:
            query: Search query
            max_results: Maximum papers to collect
            year_min: Minimum publication year
            year_max: Maximum publication year
            min_citations: Minimum citation count
            fields_of_study: List of fields to filter
            venue: Venue filter
            download_pdfs: Whether to download PDFs
            convert_markdown: Whether to convert PDFs to markdown
            download_latex: Whether to download ArXiv LaTeX source

        Returns:
            List of collected papers with local file paths
        """
        # Search for papers
        papers = self.search_papers(
            query=query,
            max_results=max_results,
            year_min=year_min,
            year_max=year_max,
            min_citations=min_citations,
            fields_of_study=fields_of_study,
            venue=venue
        )

        if not papers:
            print("No papers found.")
            return []

        # Download papers
        if download_pdfs:
            print(f"\n{'='*70}")
            print(f"DOWNLOADING PAPERS")
            print(f"{'='*70}\n")

            for i, paper in enumerate(papers, 1):
                title = paper.get("title", "Unknown")
                print(f"[{i}/{len(papers)}] {title[:60]}...")

                # Create paper folder
                paper_id = paper.get("paperId", f"unknown_{i}")
                safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_'
                                    for c in title[:50])
                paper_folder = self.run_dir / f"{safe_title}_{paper_id[:8]}"
                paper_folder.mkdir(parents=True, exist_ok=True)

                # Save metadata
                metadata_path = paper_folder / "metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(paper, f, indent=2)
                print(f"  ✓ Metadata saved")

                # Download PDF
                pdf_path = self.download_pdf(paper, paper_folder)
                paper['local_pdf_path'] = str(pdf_path) if pdf_path else None

                # Convert to markdown
                if pdf_path and convert_markdown:
                    md_path = self.convert_to_markdown(pdf_path, paper_folder)
                    paper['local_markdown_path'] = str(md_path) if md_path else None

                # Download ArXiv LaTeX source
                if download_latex:
                    external_ids = paper.get("externalIds", {})
                    arxiv_id = external_ids.get("ArXiv")
                    if arxiv_id:
                        latex_path = self.download_arxiv_source(arxiv_id, paper_folder)
                        paper['local_latex_path'] = str(latex_path) if latex_path else None
                    else:
                        print(f"  ⊘ No ArXiv ID, skipping LaTeX source")

                paper['paper_folder'] = str(paper_folder)

        # Save collection summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'max_results': max_results,
            'filters': {
                'year_min': year_min,
                'year_max': year_max,
                'min_citations': min_citations,
                'fields_of_study': fields_of_study,
                'venue': venue
            },
            'papers_found': len(papers),
            'papers_downloaded': sum(1 for p in papers if p.get('local_pdf_path')),
            'papers_converted': sum(1 for p in papers if p.get('local_markdown_path')),
            'papers_with_latex': sum(1 for p in papers if p.get('local_latex_path')),
            'papers': papers
        }

        summary_path = self.run_dir / "collection_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        # Create papers_to_process.txt for pipeline
        papers_to_process_path = self.run_dir / "papers_to_process.txt"
        with open(papers_to_process_path, 'w', encoding='utf-8') as f:
            f.write("# Papers collected for processing\n")
            f.write(f"# Collection: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Query: {query}\n")
            f.write("#\n")
            f.write("# Use with: python experimental_setup_eval/run_pipeline.py papers_to_process.txt\n")
            f.write("#\n\n")

            for paper in papers:
                paper_folder = paper.get('paper_folder')
                if paper_folder:
                    # Try to find the best markdown file
                    folder_path = Path(paper_folder)

                    # Prefer paper_clean.md (from improved extraction)
                    markdown_path = folder_path / "paper_clean.md"
                    if not markdown_path.exists():
                        # Fallback to paper.md
                        markdown_path = folder_path / "paper.md"

                    if markdown_path.exists():
                        f.write(f"{markdown_path}\n")
                    else:
                        # Comment out papers without markdown
                        title = paper.get('title', 'Unknown')[:60]
                        f.write(f"# No markdown: {title}\n")

        print(f"\n{'='*70}")
        print(f"COLLECTION SUMMARY")
        print(f"{'='*70}")
        print(f"Papers found: {len(papers)}")
        print(f"Papers downloaded: {sum(1 for p in papers if p.get('local_pdf_path'))}")
        if convert_markdown:
            print(f"Papers converted to markdown: {sum(1 for p in papers if p.get('local_markdown_path'))}")
        if download_latex:
            print(f"Papers with LaTeX source: {sum(1 for p in papers if p.get('local_latex_path'))}")
        print(f"\nOutput directory: {self.run_dir}")
        print(f"Summary saved: {summary_path}")
        print(f"Pipeline input file: {papers_to_process_path}")

        return papers


def main():
    parser = argparse.ArgumentParser(
        description="Collect research papers using Semantic Scholar API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 2 papers about transformers
  python %(prog)s --query "transformer neural networks" --max-results 2 --download-pdfs --convert-markdown

  # Collect papers with filters
  python %(prog)s --query "ResNet image classification" --year-min 2015 --min-citations 100 --max-results 2 --download-pdfs

  # Collect from specific field
  python %(prog)s --query "BERT language model" --field "Computer Science" --max-results 2 --download-pdfs --convert-markdown
        """
    )

    # API configuration
    parser.add_argument(
        "--api-key",
        type=str,
        help="Semantic Scholar API key (or set S2_API_KEY env variable)"
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=2.0,
        help="Seconds between requests (default: 2.0)"
    )

    # Search parameters
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query string"
    )
    parser.add_argument(
        "--field",
        type=str,
        help="Field of study filter (e.g., 'Computer Science', 'Machine Learning')"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=2,
        help="Maximum number of papers to collect (default: 2)"
    )
    parser.add_argument(
        "--year-min",
        type=int,
        help="Minimum publication year"
    )
    parser.add_argument(
        "--year-max",
        type=int,
        help="Maximum publication year"
    )
    parser.add_argument(
        "--min-citations",
        type=int,
        help="Minimum citation count"
    )
    parser.add_argument(
        "--venue",
        type=str,
        help="Filter by venue (e.g., 'NeurIPS', 'ICML', 'CVPR')"
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/papers",
        help="Base output directory (default: data/papers)"
    )
    parser.add_argument(
        "--download-pdfs",
        action="store_true",
        help="Download PDFs if available"
    )
    parser.add_argument(
        "--convert-markdown",
        action="store_true",
        help="Convert PDFs to markdown (requires pymupdf4llm)"
    )
    parser.add_argument(
        "--download-latex",
        action="store_true",
        help="Download ArXiv LaTeX source if available"
    )
    parser.add_argument(
        "--download-all-formats",
        action="store_true",
        help="Download all formats (PDF + Markdown + LaTeX) - convenience flag"
    )

    args = parser.parse_args()

    # Handle --download-all-formats convenience flag
    if args.download_all_formats:
        args.download_pdfs = True
        args.convert_markdown = True
        args.download_latex = True

    # Initialize collector
    collector = PaperCollector(
        api_key=args.api_key,
        throttle_seconds=args.throttle,
        base_output_dir=args.output_dir
    )

    # Build fields of study list
    fields_of_study = None
    if args.field:
        fields_of_study = [args.field]

    # Collect papers
    papers = collector.collect_papers(
        query=args.query,
        max_results=args.max_results,
        year_min=args.year_min,
        year_max=args.year_max,
        min_citations=args.min_citations,
        fields_of_study=fields_of_study,
        venue=args.venue,
        download_pdfs=args.download_pdfs,
        convert_markdown=args.convert_markdown,
        download_latex=args.download_latex
    )

    if papers:
        print(f"\n✓ Collection complete! Check {collector.run_dir} for results.")
        return 0
    else:
        print(f"\n✗ No papers collected.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
