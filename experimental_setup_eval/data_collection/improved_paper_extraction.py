"""
Improved Paper Text Extraction

Prefers cleaner sources in this order:
1. ArXiv LaTeX source (cleanest)
2. PDF with better extraction (PyPDF2)
3. Fallback to pymupdf4llm

Also includes text cleaning to remove formatting artifacts.
"""

import re
from pathlib import Path
from typing import Optional, Tuple
import subprocess
import tempfile
import unicodedata


def clean_markdown_text(text: str) -> str:
    """
    Clean markdown text from PDF/LaTeX conversion while preserving
    basic structure (headings, paragraphs, line breaks).

    The goal is to remove the worst artifacts (LaTeX commands,
    stray math symbols, double-bold markers) without collapsing
    everything into a single line or stripping useful punctuation.
    """
    # Normalize unicode and strip combining accents (e.g. stray hats)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # Remove simple LaTeX-style commands like \emph{...}, \textbf{...}
    text = re.sub(r"\\[a-zA-Z]+\{([^\}]*)\}", r"\1", text)

    # Remove math delimiters
    text = re.sub(r"\$+", "", text)

    # Strip markdown bold/italic markers but keep inner text
    text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"\*([^\*]+)\*", r"\1", text)      # *italic*
    text = re.sub(r"\_([^\_]+)\_", r"\1", text)      # _emphasis_

    # Process line-by-line to preserve paragraphs/headings
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line

        # Collapse multiple spaces/tabs inside the line
        line = re.sub(r"[ \t]+", " ", line)

        # Trim trailing spaces
        line = line.rstrip()

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned


def convert_latex_to_markdown(latex_dir: Path, output_path: Path) -> bool:
    """
    Convert LaTeX source to clean markdown using pandoc.

    Args:
        latex_dir: Directory containing LaTeX files
        output_path: Path to save markdown

    Returns:
        True if successful
    """
    # Find main .tex file
    tex_files = list(latex_dir.glob("*.tex"))

    if not tex_files:
        return False

    # Try to find main.tex or paper.tex first
    main_tex = None
    for name in ["main.tex", "paper.tex", "ms.tex"]:
        candidate = latex_dir / name
        if candidate.exists():
            main_tex = candidate
            break

    if not main_tex:
        # Use first .tex file
        main_tex = tex_files[0]

    try:
        # Use pandoc to convert LaTeX to markdown
        subprocess.run(
            [
                "pandoc",
                str(main_tex),
                "-f", "latex",
                "-t", "markdown",
                "-o", str(output_path),
                "--wrap=none",  # Don't wrap lines
                "--strip-comments"  # Remove LaTeX comments
            ],
            check=True,
            capture_output=True,
            timeout=60
        )

        print(f"  ✓ Converted LaTeX to markdown using pandoc")

        # Clean the output
        with open(output_path, 'r', encoding='utf-8') as f:
            text = f.read()

        cleaned = clean_markdown_text(text)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)

        return True

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  ⊘ Pandoc conversion failed: {e}")
        return False


def extract_text_from_pdf_pypdf(pdf_path: Path, output_path: Path) -> bool:
    """
    Extract text from PDF using PyPDF2 (better for text-based PDFs).

    Args:
        pdf_path: Path to PDF file
        output_path: Path to save markdown

    Returns:
        True if successful
    """
    try:
        import PyPDF2

        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)

            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

            full_text = "\n\n".join(text_parts)

            # Clean the text
            cleaned = clean_markdown_text(full_text)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)

            print(f"  ✓ Extracted text using PyPDF2 ({len(cleaned)} chars)")
            return True

    except ImportError:
        print(f"  ⊘ PyPDF2 not installed")
        return False
    except Exception as e:
        print(f"  ⊘ PyPDF2 extraction failed: {e}")
        return False


def get_best_paper_markdown(paper_folder: Path) -> Tuple[Optional[Path], str]:
    """
    Get the best available markdown for a paper.

    Tries in order:
    1. LaTeX source → markdown (cleanest)
    2. PDF → PyPDF2 → markdown (good for text PDFs)
    3. PDF → pymupdf4llm → markdown (fallback)
    4. Existing paper.md

    Args:
        paper_folder: Folder containing paper files

    Returns:
        (markdown_path, source_type) tuple
    """
    pdf_path = paper_folder / "paper.pdf"
    latex_dir = paper_folder / "latex_source"
    existing_md = paper_folder / "paper.md"
    improved_md = paper_folder / "paper_clean.md"

    # Try LaTeX first
    if latex_dir.exists():
        print(f"  → Attempting LaTeX → markdown conversion...")
        if convert_latex_to_markdown(latex_dir, improved_md):
            return improved_md, "latex"

    # Try PDF with PyPDF2
    if pdf_path.exists():
        print(f"  → Attempting PDF → PyPDF2 extraction...")
        if extract_text_from_pdf_pypdf(pdf_path, improved_md):
            return improved_md, "pypdf2"

    # Fallback to existing markdown (already created by pymupdf4llm)
    if existing_md.exists():
        print(f"  → Using existing pymupdf4llm markdown")

        # Clean it
        with open(existing_md, 'r', encoding='utf-8') as f:
            text = f.read()

        cleaned = clean_markdown_text(text)

        with open(improved_md, 'w', encoding='utf-8') as f:
            f.write(cleaned)

        return improved_md, "pymupdf_cleaned"

    return None, "none"


def improve_paper_collection(base_dir: Path):
    """
    Post-process collected papers to get better markdown.

    Args:
        base_dir: Base directory containing paper collection folders
    """
    print("\n" + "="*70)
    print("IMPROVING PAPER TEXT EXTRACTION")
    print("="*70 + "\n")

    # Find all paper folders
    paper_folders = []
    for item in base_dir.rglob("*/"):
        if (item / "paper.pdf").exists() or (item / "latex_source").exists():
            paper_folders.append(item)

    if not paper_folders:
        print("No paper folders found")
        return

    print(f"Found {len(paper_folders)} paper folders\n")

    for i, folder in enumerate(paper_folders, 1):
        print(f"[{i}/{len(paper_folders)}] {folder.name}")

        improved_md, source = get_best_paper_markdown(folder)

        if improved_md:
            print(f"  ✓ Created clean markdown from: {source}")
            print(f"    {improved_md}")
        else:
            print(f"  ✗ No sources available")

        print()

    print("="*70)
    print("✓ Paper improvement complete")
    print("="*70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python improved_paper_extraction.py <collection_dir>")
        print("Example: python improved_paper_extraction.py data/papers/20241117_120000_collection/")
        sys.exit(1)

    base_dir = Path(sys.argv[1])

    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)

    improve_paper_collection(base_dir)
