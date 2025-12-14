"""
OmegaConf-based configuration loader for experimental setup evaluation.
Follows AI Scientist patterns with typed validation and CLI support.
"""

import os
from pathlib import Path
from typing import Optional, cast

from omegaconf import OmegaConf
from dotenv import load_dotenv

# Handle imports for both package and standalone usage
# Handle imports for both package and standalone usage
try:
    from .schema import EvalSetupConfig
except ImportError:
    # Fallback for standalone usage - add current directory to path
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from schema import EvalSetupConfig


class ConfigLoader:
    """OmegaConf-based configuration loader with validation."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to config file. If None, uses default config.yaml.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        self.config_path = Path(config_path)
        self._config: Optional[EvalSetupConfig] = None

        # Load .env file from current directory (experimental_setup_eval)
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            
        # Also try parent directory for backward compatibility
        parent_env = Path(__file__).parent.parent / ".env"
        if parent_env.exists():
            load_dotenv(parent_env)

    def load(self, use_cli_args: bool = False) -> EvalSetupConfig:
        """
        Load and validate configuration.

        Args:
            use_cli_args: Whether to merge CLI arguments

        Returns:
            Fully validated configuration
        """
        if self._config is None:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")

            # Load YAML configuration
            cfg = OmegaConf.load(self.config_path)

            # Merge CLI args if requested
            if use_cli_args:
                cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

            # Create schema for validation
            schema = OmegaConf.structured(EvalSetupConfig)

            # Merge schema with config for validation
            self._config = cast(EvalSetupConfig, OmegaConf.merge(schema, cfg))

            # Validate quality weights sum to 1.0
            weights = self._config.evaluation.quality_weights
            total = weights.completeness + weights.diversity + weights.alignment + weights.datasets + weights.protocol
            if abs(total - 1.0) > 0.001:
                raise ValueError(f"Quality weights must sum to 1.0, got {total}")

        return self._config

    def get_model(self, model_name: Optional[str] = None) -> str:
        """Get model configuration with alias resolution."""
        config = self.load()

        if model_name is None:
            return config.models.default

        # Check if it's an alias
        if model_name in config.models.available:
            return config.models.available[model_name]

        # Return as-is if not found in aliases
        return model_name

    def check_model_availability(self, model_name: str) -> bool:
        """Check if model is available based on API keys."""
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        has_groq = bool(os.getenv("GROQ_API_KEY"))

        model_lower = model_name.lower()

        if any(provider in model_lower for provider in ["gpt", "openai"]):
            return has_openai
        elif any(provider in model_lower for provider in ["claude", "anthropic"]):
            return has_anthropic
        elif any(provider in model_lower for provider in ["llama", "groq"]):
            return has_groq
        else:
            return has_openai or has_anthropic or has_groq

    def get_available_model_with_key(self) -> str:
        """Get first available model that has valid API key."""
        config = self.load()

        # Try default first
        if self.check_model_availability(config.models.default):
            return config.models.default

        # Try all available models
        for model_name in config.models.available.values():
            if self.check_model_availability(model_name):
                return model_name

        raise RuntimeError(
            "No API keys found. Please set one of: GROQ_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY"
        )


def load_config(config_path: Optional[str] = None, use_cli_args: bool = False) -> EvalSetupConfig:
    """Load configuration with validation."""
    return ConfigLoader(config_path).load(use_cli_args)


# Global config instance
_default_config: Optional[ConfigLoader] = None

def get_default_config() -> ConfigLoader:
    """Get the default global config instance."""
    global _default_config
    if _default_config is None:
        _default_config = ConfigLoader()
    return _default_config