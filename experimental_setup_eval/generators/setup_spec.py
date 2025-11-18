"""
Experimental Setup Function Specification

Defines the structured output schema for experimental setup generation.
"""

from ai_scientist.treesearch.backend import FunctionSpec


experimental_setup_spec = FunctionSpec(
    name="generate_experimental_setup",
    description="Generate a comprehensive experimental setup for evaluating a research paper",
    json_schema={
        "type": "object",
        "properties": {
            "baselines": {
                "type": "array",
                "description": "List of baseline methods to compare against",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the baseline method"
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief description of the baseline"
                        },
                        "complexity": {
                            "type": "string",
                            "enum": ["simple", "moderate", "complex"],
                            "description": "Complexity level of the baseline"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this baseline is important"
                        }
                    },
                    "required": ["name", "description", "complexity", "rationale"]
                }
            },
            "metrics": {
                "type": "array",
                "description": "Evaluation metrics to measure performance",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the metric"
                        },
                        "description": {
                            "type": "string",
                            "description": "What the metric measures"
                        },
                        "type": {
                            "type": "string",
                            "enum": ["primary", "diagnostic", "auxiliary"],
                            "description": "Type of metric (primary, diagnostic, or auxiliary)"
                        },
                        "higher_is_better": {
                            "type": "boolean",
                            "description": "Whether higher values are better"
                        },
                        "expected_range": {
                            "type": "string",
                            "description": "Expected range of values (e.g., '0-1', '0-100')"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this metric is appropriate"
                        }
                    },
                    "required": ["name", "description", "type", "higher_is_better", "expected_range", "rationale"]
                }
            },
            "datasets": {
                "type": "array",
                "description": "Datasets to use for evaluation",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the dataset"
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of the dataset"
                        },
                        "size": {
                            "type": "string",
                            "description": "Dataset size (e.g., '10k samples', '1M tokens')"
                        },
                        "domain": {
                            "type": "string",
                            "description": "Domain or field (e.g., 'NLP', 'CV', 'general')"
                        },
                        "split": {
                            "type": "string",
                            "description": "Train/val/test split information"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this dataset is appropriate"
                        }
                    },
                    "required": ["name", "description", "size", "domain", "split", "rationale"]
                }
            },
            "significance_tests": {
                "type": "array",
                "description": "Statistical tests for validating results",
                "items": {
                    "type": "object",
                    "properties": {
                        "test_name": {
                            "type": "string",
                            "description": "Name of the statistical test"
                        },
                        "when_to_use": {
                            "type": "string",
                            "description": "When this test should be applied"
                        },
                        "assumptions": {
                            "type": "string",
                            "description": "Statistical assumptions required"
                        }
                    },
                    "required": ["test_name", "when_to_use", "assumptions"]
                }
            },
            "experimental_protocol": {
                "type": "object",
                "description": "Detailed experimental procedure",
                "properties": {
                    "num_runs": {
                        "type": "integer",
                        "description": "Number of experimental runs"
                    },
                    "random_seeds": {
                        "type": "string",
                        "description": "Random seed strategy (e.g., '5 different seeds')"
                    },
                    "validation_strategy": {
                        "type": "string",
                        "description": "Validation approach (e.g., 'k-fold cross-validation', 'hold-out')"
                    },
                    "hyperparameter_tuning": {
                        "type": "string",
                        "description": "Hyperparameter tuning approach"
                    },
                    "compute_budget": {
                        "type": "string",
                        "description": "Estimated computational resources needed"
                    },
                    "reproducibility_notes": {
                        "type": "string",
                        "description": "Additional notes for reproducibility"
                    }
                },
                "required": ["num_runs", "random_seeds", "validation_strategy", "hyperparameter_tuning", "compute_budget"]
            },
            "reasoning": {
                "type": "string",
                "description": "Overall rationale for why this experimental setup is appropriate for evaluating the proposed research"
            }
        },
        "required": ["baselines", "metrics", "datasets", "significance_tests", "experimental_protocol", "reasoning"]
    }
)
