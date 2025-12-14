"""
Enhanced Ground Truth Extractor with Domain Awareness

Improvements over original:
1. Multi-stage extraction with domain-specific refinement
2. Domain terminology detection and handling
3. Better parsing of technical terms and variants
4. Confidence scoring based on extraction quality
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ai_scientist.llm import create_client, get_response_from_llm


@dataclass
class ExtractionResult:
    """Structured result from extraction."""
    baselines: List[Dict[str, str]]
    metrics: List[Dict[str, str]]
    datasets: List[Dict[str, str]]
    has_experiments: bool
    extraction_confidence: str
    domain: str
    technical_terms: List[str]
    extraction_method: str = "enhanced"


class EnhancedGroundTruthExtractor:
    """
    Enhanced extractor with domain awareness and multi-stage processing.

    Features:
    - Domain detection and specialized extraction
    - Technical terminology identification
    - Variant/synonym detection for experimental components
    - Multi-stage validation and refinement
    """

    def __init__(self, model: str = "gpt-4o-2024-11-20"):
        """Initialize enhanced extractor."""
        self.model = model
        self.client, self.model_name = create_client(model)

        # Domain-specific keywords for detection
        self.domain_keywords = {
            "machine_learning": ["neural network", "deep learning", "transformer", "CNN", "RNN", "LSTM"],
            "nlp": ["language model", "NLP", "text processing", "tokenization", "embedding"],
            "computer_vision": ["image classification", "object detection", "CNN", "vision", "image"],
            "chemistry": ["molecular", "chemical", "reaction", "synthesis", "compound"],
            "biology": ["protein", "gene", "biological", "sequence", "genomic"],
            "physics": ["quantum", "particle", "simulation", "physics", "mechanical"]
        }

    def extract_from_paper(self, paper_markdown: str, paper_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Enhanced extraction with domain awareness and multi-stage processing.

        Args:
            paper_markdown: Full paper text
            paper_context: Optional context (title, abstract, etc.)

        Returns:
            Enhanced extraction result with domain information
        """
        try:
            # Stage 1: Detect domain
            domain = self._detect_domain(paper_markdown, paper_context)

            # Stage 2: General extraction (baseline approach)
            general_result = self._extract_general(paper_markdown)

            # Stage 3: Domain-specific refinement
            domain_result = self._extract_domain_specific(paper_markdown, domain, general_result)

            # Stage 4: Technical term extraction and validation
            final_result = self._refine_and_validate(paper_markdown, domain_result, domain)

            return final_result.to_dict()

        except Exception as e:
            print(f"Warning: Enhanced extraction failed: {e}")
            return self._fallback_extraction(paper_markdown)

    def _detect_domain(self, paper_text: str, paper_context: Optional[Dict] = None) -> str:
        """Detect the research domain of the paper."""
        text_to_analyze = paper_text[:3000]  # Use beginning of paper

        if paper_context:
            text_to_analyze = f"{paper_context.get('title', '')} {paper_context.get('abstract', '')} {text_to_analyze}"

        text_lower = text_to_analyze.lower()
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score

        detected_domain = max(domain_scores, key=domain_scores.get) if domain_scores else "general"
        return detected_domain

    def _extract_general(self, paper_text: str) -> Dict[str, Any]:
        """General extraction using existing approach."""
        prompt = self._create_general_extraction_prompt(paper_text)

        system_message = """You are an expert at extracting experimental setup details from research papers.
Focus on finding the actual experimental components used, not proposed methods."""

        try:
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.model_name,
                system_message=system_message,
                temperature=0.2
            )
            return self._parse_json_response(response)
        except Exception as e:
            return {"baselines": [], "metrics": [], "datasets": [], "error": str(e)}

    def _extract_domain_specific(self, paper_text: str, domain: str, general_result: Dict) -> Dict[str, Any]:
        """Domain-specific extraction to catch technical terminology."""
        domain_prompts = {
            "machine_learning": """
Extract ML-specific experimental details:
- Model architectures (CNN, RNN, Transformer variants, etc.)
- Training techniques and baselines
- Performance metrics (accuracy, F1, BLEU, perplexity, etc.)
- Standard ML datasets (ImageNet, MNIST, GLUE, etc.)
Pay special attention to:
- Model size indicators (B parameters, layers)
- Technical model names with prefixes/suffixes
- Domain-specific evaluation metrics
""",
            "nlp": """
Extract NLP-specific experimental details:
- Language models and variants (GPT, BERT, T5, etc.)
- NLP evaluation metrics (BLEU, ROUGE, BERTScore, etc.)
- Text datasets and benchmarks (GLUE, SuperGLUE, etc.)
- Text processing techniques
""",
            "chemistry": """
Extract chemistry-specific experimental details:
- Chemical synthesis methods and techniques
- Molecular property prediction models
- Chemical datasets (USPTO, QM9, etc.)
- Chemistry-specific metrics (yield, selectivity, accuracy)
""",
            "general": """
Extract any experimental components missed in the general extraction:
- Additional baseline methods or variants
- Specialized metrics or evaluation approaches
- Domain-specific datasets
- Technical terminology that may have been missed
"""
        }

        domain_prompt = domain_prompts.get(domain, domain_prompts["general"])

        prompt = f"""
Based on this research paper, {domain_prompt}

Look for experimental components that may have been missed in this initial extraction:
{json.dumps(general_result, indent=2)}

Paper text:
{paper_text[:12000]}

Provide additional components in the same JSON format:
{{
    "additional_baselines": [list of missed baseline methods],
    "additional_metrics": [list of missed evaluation metrics],
    "additional_datasets": [list of missed datasets],
    "technical_terms": [list of domain-specific technical terms found],
    "confidence": "high/medium/low"
}}
"""

        try:
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.model_name,
                system_message=f"You are a {domain} expert extracting experimental details.",
                temperature=0.3
            )
            return self._parse_json_response(response)
        except Exception as e:
            return {"additional_baselines": [], "additional_metrics": [], "additional_datasets": [], "error": str(e)}

    def _refine_and_validate(self, paper_text: str, domain_result: Dict, domain: str) -> ExtractionResult:
        """Combine and validate extraction results."""
        # Combine general and domain-specific results
        all_baselines = (
            domain_result.get("baselines", []) +
            domain_result.get("additional_baselines", [])
        )
        all_metrics = (
            domain_result.get("metrics", []) +
            domain_result.get("additional_metrics", [])
        )
        all_datasets = (
            domain_result.get("datasets", []) +
            domain_result.get("additional_datasets", [])
        )

        # Remove duplicates while preserving structure
        unique_baselines = self._deduplicate_components(all_baselines)
        unique_metrics = self._deduplicate_components(all_metrics)
        unique_datasets = self._deduplicate_components(all_datasets)

        # Extract technical terms
        technical_terms = domain_result.get("technical_terms", [])

        # Determine confidence
        has_experiments = len(unique_baselines) > 0 or len(unique_metrics) > 0
        if len(unique_baselines) >= 2 and len(unique_metrics) >= 1:
            confidence = "high"
        elif len(unique_baselines) >= 1 or len(unique_metrics) >= 1:
            confidence = "medium"
        else:
            confidence = "low"

        return ExtractionResult(
            baselines=unique_baselines,
            metrics=unique_metrics,
            datasets=unique_datasets,
            has_experiments=has_experiments,
            extraction_confidence=confidence,
            domain=domain,
            technical_terms=technical_terms
        )

    def _deduplicate_components(self, components: List[Dict]) -> List[Dict]:
        """Remove duplicate components based on name similarity."""
        if not components:
            return []

        unique_components = []
        seen_names = set()

        for component in components:
            name = component.get("name", "").lower().strip()
            if name and name not in seen_names:
                seen_names.add(name)
                unique_components.append(component)

        return unique_components

    def _create_general_extraction_prompt(self, paper_text: str) -> str:
        """Create general extraction prompt."""
        # Truncate if too long, focusing on experimental sections
        if len(paper_text) > 15000:
            paper_text = self._extract_relevant_sections(paper_text)

        return f"""
Extract the experimental setup details from this research paper.

Focus on the ACTUAL experiments conducted, not proposed methods.

Find:
1. Baselines: Methods/models compared against
2. Metrics: Evaluation metrics used to measure performance
3. Datasets: Datasets used for evaluation

Paper text:
{paper_text}

Respond with valid JSON:
{{
    "baselines": [
        {{"name": "baseline name", "description": "brief description"}},
        ...
    ],
    "metrics": [
        {{"name": "metric name", "description": "what it measures"}},
        ...
    ],
    "datasets": [
        {{"name": "dataset name", "description": "dataset description"}},
        ...
    ]
}}
"""

    def _extract_relevant_sections(self, paper_text: str) -> str:
        """Extract experimental sections from long papers."""
        sections = ["experiment", "evaluation", "result", "method", "setup", "implementation"]
        relevant_parts = []

        lines = paper_text.split('\n')
        current_section = []
        in_relevant = False

        for line in lines:
            line_lower = line.lower()

            # Check if this is a section header
            if any(f'# {sec}' in line_lower or f'## {sec}' in line_lower for sec in sections):
                if current_section and in_relevant:
                    relevant_parts.extend(current_section)
                current_section = [line]
                in_relevant = True
            elif in_relevant:
                current_section.append(line)
                if len('\n'.join(relevant_parts + current_section)) > 12000:
                    break

        if in_relevant:
            relevant_parts.extend(current_section)

        return '\n'.join(relevant_parts) if relevant_parts else paper_text[:12000]

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: try to parse partial response
            return {"baselines": [], "metrics": [], "datasets": [], "error": "JSON parse error"}

    def _fallback_extraction(self, paper_text: str) -> Dict[str, Any]:
        """Fallback to basic extraction if enhanced method fails."""
        return {
            "baselines": [],
            "metrics": [],
            "datasets": [],
            "has_experiments": False,
            "extraction_confidence": "low",
            "domain": "unknown",
            "technical_terms": [],
            "extraction_method": "fallback",
            "error": "Enhanced extraction failed, using fallback"
        }


# Extension methods for ExtractionResult
def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary format."""
    return {
        "baselines": self.baselines,
        "metrics": self.metrics,
        "datasets": self.datasets,
        "has_experiments": self.has_experiments,
        "extraction_confidence": self.extraction_confidence,
        "domain": self.domain,
        "technical_terms": self.technical_terms,
        "extraction_method": self.extraction_method
    }

# Monkey patch the method
ExtractionResult.to_dict = to_dict