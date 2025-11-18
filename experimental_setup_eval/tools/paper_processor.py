"""
Paper Processor Tool

Extracts and processes research paper content, with ability to redact
experimental sections for experimental design evaluation.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.llm import create_client, get_response_from_llm


class PaperProcessorTool(BaseTool):
    """
    Tool for processing research papers: parsing, section extraction, and redaction.

    Supports markdown papers (from PDF-to-markdown conversion).
    """

    def __init__(self, model: str = "gpt-4o-2024-11-20"):
        """
        Initialize paper processor.

        Args:
            model: LLM model for extraction tasks (research question, domain classification)
        """
        super().__init__(
            name="paper_processor",
            description="Extract and process research paper content with optional redaction of experimental sections",
            parameters=[
                {
                    "name": "paper_path",
                    "type": "str",
                    "description": "Path to paper markdown file"
                },
                {
                    "name": "redact_sections",
                    "type": "list",
                    "description": "List of section keywords to redact (default: experiments, results, evaluation)"
                }
            ]
        )
        self.client, self.model_name = create_client(model)
        self.model = self.model_name  # Keep for compatibility

    def use_tool(
        self,
        paper_path: str,
        redact_sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process paper and extract structured content.

        Args:
            paper_path: Path to markdown paper
            redact_sections: Section keywords to redact (default: ["experiment", "result", "evaluation"])

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
                    "evaluation": str,
                    "implementation": str,
                    "raw_experimental_content": str
                },
                "paper_context": str  # Combined non-experimental context
            }
        """
        if redact_sections is None:
            redact_sections = ["experiment", "result", "evaluation"]

        paper_path = Path(paper_path)

        # Read markdown
        with open(paper_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        # Parse sections
        sections = self._parse_sections(markdown_content)

        # Extract title
        title = self._extract_title(sections)

        # Helper to find section by keyword
        def find_section(keyword: str, max_length: Optional[int] = None) -> str:
            # First try exact match in section names
            for section_name, content in sections.items():
                if keyword in section_name.lower():
                    if max_length:
                        return content[:max_length]
                    return content
            
            # For abstract: if no "abstract" section, look in header for paragraph after title
            if keyword == "abstract" and "header" in sections:
                header = sections["header"]
                lines = [l.strip() for l in header.split('\n') if l.strip()]
                # Skip date and title lines, take first substantial paragraph
                in_abstract = False
                abstract_lines = []
                for line in lines:
                    # Skip dates and titles
                    if re.match(r'^_?\d{4}-\d{2}-\d{2}_?$', line):
                        continue
                    if line.startswith('#') or line.startswith('**'):
                        in_abstract = True
                        continue
                    if in_abstract and len(line) > 50:  # Substantial text
                        abstract_lines.append(line)
                        if len(' '.join(abstract_lines)) > 500:  # Got enough
                            break
                if abstract_lines:
                    result = ' '.join(abstract_lines)
                    return result[:max_length] if max_length else result
            
            return ""

        # Extract non-experimental sections (for AI context)
        abstract = find_section("abstract", 1000)
        introduction = find_section("introduction", 1500)
        related_work = find_section("related work", 1000)
        method_description = (
            find_section("method", 1500) or
            find_section("approach", 1500) or
            find_section("model", 1500)
        )

        # Extract experimental sections (ground truth)
        experimental_sections = []
        redacted_content = {}

        for section_name, content in sections.items():
            if any(keyword in section_name.lower() for keyword in redact_sections):
                experimental_sections.append(content)

        redacted_content = {
            "experiments_section": find_section("experiment"),
            "results_section": find_section("result"),
            "evaluation_section": find_section("evaluation"),
            "implementation_section": find_section("implementation"),
            "raw_experimental_content": "\n\n".join(experimental_sections)
        }

        # Build paper context (non-experimental)
        paper_context = f"""# {title}

## Abstract
{abstract}

## Introduction
{introduction}

## Related Work
{related_work}

## Method
{method_description}
"""

        # Extract research question using LLM
        research_question = self._extract_research_question(paper_context, title)

        # Classify domain using LLM
        domain = self._classify_domain(paper_context, title, abstract)

        return {
            "title": title,
            "abstract": abstract,
            "introduction": introduction,
            "related_work": related_work,
            "method_description": method_description,
            "research_question": research_question,
            "domain": domain,
            "redacted_sections": redacted_content,
            "paper_context": paper_context,
            "full_sections": sections  # For debugging
        }

    def _clean_markdown(self, content: str) -> str:
        """
        Clean markdown content by removing HTML tags and normalizing formatting.
        
        PDF-to-markdown tools often produce HTML tags instead of pure markdown.
        This method cleans those up before section parsing.
        
        Args:
            content: Raw markdown content (possibly with HTML)
            
        Returns:
            Cleaned markdown string
        """
        import re
        
        # Remove HTML figure tags (common in academic papers)
        content = re.sub(r'<figure[^>]*>.*?</figure>', '', content, flags=re.DOTALL)
        content = re.sub(r'<embed[^>]*>', '', content)
        content = re.sub(r'<figcaption>.*?</figcaption>', '', content, flags=re.DOTALL)
        
        # Remove other common HTML tags
        content = re.sub(r'<span[^>]*>.*?</span>', lambda m: m.group(0).replace('<span', '').replace('</span>', ''), content, flags=re.DOTALL)
        content = re.sub(r'<div[^>]*>.*?</div>', '', content, flags=re.DOTALL)
        
        # Convert HTML headers to markdown
        content = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1', content, flags=re.DOTALL)
        content = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1', content, flags=re.DOTALL)
        content = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1', content, flags=re.DOTALL)
        
        # Clean up LaTeX/PDF artifacts
        content = re.sub(r'data-latex-[a-z-]+="[^"]*"', '', content)
        content = re.sub(r'id="[^"]*"', '', content)
        
        # Clean up extra whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'[ \t]+\n', '\n', content)
        
        return content

    def _parse_sections(self, markdown_content: str) -> Dict[str, str]:
        """
        Parse markdown into sections with robust pattern matching.

        Handles markdown headers (#), bold section names (**Section**),
        and HTML tags after cleaning.
        """
        # Clean HTML tags first
        markdown_content = self._clean_markdown(markdown_content)
        
        sections = {}
        current_section = "header"
        sections[current_section] = []

        for line in markdown_content.split('\n'):
            # Check for markdown headers (# or ##)
            if re.match(r'^#{1,4}\s+.+', line):
                current_section = re.sub(r'^#{1,4}\s+', '', line).strip().lower()
                sections[current_section] = []
            # Check for bold section headers (common in PDF-to-markdown)
            elif line.strip().startswith('**') and line.strip().endswith('**') and len(line.strip()) < 100:
                potential_section = line.strip().strip('*').strip().lower()
                # Common section names
                if any(keyword in potential_section for keyword in [
                    'abstract', 'introduction', 'method', 'experiment', 'result',
                    'evaluation', 'conclusion', 'related work', 'background',
                    'approach', 'model', 'dataset', 'implementation', 'training'
                ]):
                    current_section = potential_section
                    sections[current_section] = []
                else:
                    sections[current_section].append(line)
            else:
                sections[current_section].append(line)

        # Join sections
        for section in sections:
            sections[section] = '\n'.join(sections[section])

        return sections

    def _extract_title(self, sections: Dict[str, str]) -> str:
        """
        Extract paper title from sections.
        
        Looks for the first substantial text that's likely a title,
        filtering out dates, metadata, and short fragments.
        """
        # First check if title is a section name (common when title is in bold)
        for section_name in sections.keys():
            if len(section_name) > 20 and 'introduction' not in section_name.lower():
                # This might be the title
                cleaned = section_name.strip('*#').strip()
                if not re.match(r'^_?\d{4}-\d{2}-\d{2}_?$', cleaned):
                    return cleaned
        
        # Look in header section
        header = sections.get("header", "")

        # Try to find title (usually first non-empty, substantial line)
        lines = [l.strip() for l in header.split('\n') if l.strip()]
        
        for line in lines[:15]:  # Check first 15 lines
            # Clean up markdown formatting
            cleaned = line.strip('*#').strip()
            
            # Skip non-title content
            if not cleaned:
                continue
            if re.match(r'^_?\d{4}-\d{2}-\d{2}_?$', cleaned):  # Date pattern
                continue
            if len(cleaned) < 10:  # Too short to be a title
                continue
            if cleaned.startswith('[') and cleaned.endswith(']'):  # Link
                continue
            # If it starts with # or **, it's likely the title
            if line.startswith('#') or line.startswith('**'):
                return cleaned[:200]  # Found it
            # If it's after we've seen a date, and it's substantial, it's the title
            if len(cleaned) > 20:
                return cleaned[:200]

        # Fallback: try to find title in any section header
        for section_name in sections.keys():
            if 'title' in section_name.lower():
                content = sections[section_name].strip()
                if content and len(content) > 10:
                    return content[:200]  # Truncate if too long

        return "Unknown Title"

    def _extract_research_question(self, paper_context: str, title: str) -> str:
        """
        Use LLM to extract research question/hypothesis from paper.

        Args:
            paper_context: Non-experimental paper content
            title: Paper title

        Returns:
            Extracted research question
        """
        prompt = f"""You are an expert researcher. Given this paper's context, extract the main research question or hypothesis.

# Paper Title
{title}

# Paper Context
{paper_context[:2000]}

# Task
Identify and state the main research question or hypothesis in 1-2 sentences. Focus on what the paper is trying to answer or prove.

Response format: Just the research question/hypothesis, nothing else."""

        try:
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.model,
                system_message="You are an expert researcher.",
                temperature=0.3
            )
            return response.strip()
        except Exception as e:
            print(f"Warning: Could not extract research question: {e}")
            return "Research question extraction failed"

    def _classify_domain(self, paper_context: str, title: str, abstract: str) -> str:
        """
        Use LLM to classify paper domain.

        Args:
            paper_context: Non-experimental paper content
            title: Paper title
            abstract: Paper abstract

        Returns:
            Domain classification (CV, NLP, RL, Theory, Systems, etc.)
        """
        prompt = f"""You are an expert ML researcher. Classify this paper into ONE of these domains:

Domains:
- CV (Computer Vision)
- NLP (Natural Language Processing)
- RL (Reinforcement Learning)
- ML-Theory (Machine Learning Theory)
- ML-Systems (ML Systems/Infrastructure)
- Multimodal (Vision-Language, etc.)
- Other-ML (Other ML domains)

# Paper Title
{title}

# Abstract
{abstract[:500]}

# Task
Respond with ONLY the domain code (e.g., "CV" or "NLP"), nothing else."""

        try:
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.model,
                system_message="You are an expert ML researcher.",
                temperature=0.1
            )
            domain = response.strip().upper()

            # Validate
            valid_domains = ["CV", "NLP", "RL", "ML-THEORY", "ML-SYSTEMS", "MULTIMODAL", "OTHER-ML"]
            if domain in valid_domains:
                return domain
            else:
                print(f"Warning: Invalid domain '{domain}', defaulting to OTHER-ML")
                return "OTHER-ML"
        except Exception as e:
            print(f"Warning: Could not classify domain: {e}")
            return "OTHER-ML"
