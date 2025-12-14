"""
Native Anthropic Tool Use Backend

Implements function calling for Anthropic models using their native tool use API.
Maintains exact same interface as AI Scientist backends while leveraging Claude's
native tool calling capabilities for better reliability and performance.
"""

import json
import time
import logging
from typing import Tuple, Dict, Any, Optional

from ai_scientist.treesearch.backend.utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from ai_scientist.treesearch.backend import backend_anthropic
from ai_scientist.llm import create_client
import anthropic

logger = logging.getLogger("ai-scientist")

# Extend FunctionSpec with Anthropic-specific methods
def as_anthropic_tool_dict(self) -> Dict[str, Any]:
    """Convert FunctionSpec to Anthropic tool format."""
    return {
        "name": self.name,
        "description": self.description,
        "input_schema": self.json_schema
    }

def anthropic_tool_choice_dict(self) -> Dict[str, Any]:
    """Create Anthropic tool choice to force tool use."""
    return {
        "type": "tool",
        "name": self.name
    }

# Monkey patch the FunctionSpec class to add Anthropic support
FunctionSpec.as_anthropic_tool_dict = property(as_anthropic_tool_dict)
FunctionSpec.anthropic_tool_choice_dict = property(anthropic_tool_choice_dict)


def native_anthropic_query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> Tuple[OutputType, float, int, int, Dict]:
    """
    Enhanced Anthropic query with native tool use support.

    Uses Anthropic's native tool calling API instead of function call simulation.
    Maintains exact same interface as AI Scientist backends.

    Args:
        system_message: System message
        user_message: User message
        func_spec: Function specification for tool calling
        **model_kwargs: Model parameters

    Returns:
        Tuple of (output, request_time, input_tokens, output_tokens, info)
    """

    if func_spec is None:
        # No tool calling - use original implementation
        return backend_anthropic.query(system_message, user_message, None, **model_kwargs)

    # Native tool calling implementation - use direct Anthropic client from llm.py (not Bedrock)
    model_name = model_kwargs.get("model", "claude-3-5-sonnet-20241022")
    client, _ = create_client(model_name)  # This uses direct Anthropic API for "claude-" models

    filtered_kwargs: Dict = {k: v for k, v in model_kwargs.items() if v is not None}
    # Handle max_tokens
    if "haiku" in model_name.lower():
        # Haiku has a strict 4096 output token limit
        current_max = filtered_kwargs.get("max_tokens", 8192)
        filtered_kwargs["max_tokens"] = min(current_max, 4096)
    elif "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 8192

    # Set up native tool calling
    filtered_kwargs["tools"] = [func_spec.as_anthropic_tool_dict]
    filtered_kwargs["tool_choice"] = func_spec.anthropic_tool_choice_dict

    # Handle Anthropic system message requirements
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_message)

    t0 = time.time()
    try:
        message = backoff_create(
            client.messages.create,
            backend_anthropic.ANTHROPIC_TIMEOUT_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )
    except Exception as e:
        logger.error(f"Anthropic native tool calling failed: {e}")
        raise

    req_time = time.time() - t0

    # Parse tool use response
    try:
        output = _parse_native_tool_response(message, func_spec)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse tool use response from message: {message}")
        logger.error(f"Parse error: {e}")
        raise json.JSONDecodeError(f"Invalid tool use response: {e}", str(message), 0)

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
        "tool_call_method": "anthropic_native",
    }

    return output, req_time, in_tokens, out_tokens, info


def _parse_native_tool_response(message, func_spec: FunctionSpec) -> Dict[str, Any]:
    """
    Parse native tool use response from Anthropic.

    Anthropic returns tool_use blocks in message.content when tools are used.
    """
    tool_use_content = None

    # Find tool_use content block
    for content_block in message.content:
        if hasattr(content_block, 'type') and content_block.type == "tool_use":
            if content_block.name == func_spec.name:
                tool_use_content = content_block
                break

    if tool_use_content is None:
        # Check if we got text instead (fallback scenario)
        text_content = None
        for content_block in message.content:
            if hasattr(content_block, 'type') and content_block.type == "text":
                text_content = content_block.text
                break

        if text_content:
            logger.warning(f"Expected tool_use but got text response: {text_content}")
            # Try to parse as JSON in case model returned JSON in text
            try:
                return json.loads(text_content.strip())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"No tool_use content found for function '{func_spec.name}' in response")

    # Extract tool input - Anthropic returns this as already-parsed dict
    tool_input = tool_use_content.input

    if not isinstance(tool_input, dict):
        raise ValueError(f"Tool input is not a dict: {type(tool_input)} = {tool_input}")

    # Optional: Validate against schema
    try:
        import jsonschema
        jsonschema.validate(tool_input, func_spec.json_schema)
        logger.debug(f"Tool use response validated successfully: {func_spec.name}")
    except ImportError:
        logger.debug("jsonschema not available for validation")
    except jsonschema.ValidationError as e:
        logger.debug(f"Tool use response validation failed for {func_spec.name}: {e}")
        # Don't fail on validation errors - just warn

    return tool_input


def smart_query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    Smart backend that routes to appropriate implementation.

    - Groq models: Use Groq native function calling
    - Anthropic models: Use native tool calling
    - OpenAI models: Use native function calling
    - Other models: Attempt original, fallback to Anthropic-style if needed

    Args:
        system_message: System message
        user_message: User message
        func_spec: Function specification for tool/function calling
        **model_kwargs: Model parameters including 'model' name

    Returns:
        OutputType: Just the output (string or dict), not the full tuple
    """
    model = model_kwargs.get("model", "").lower()

    # Route based on model provider
    if any(provider in model for provider in ["llama", "groq", "mixtral", "gemma"]):
        # Use Groq API
        logger.debug(f"Using Groq API for model: {model}")
        try:
            from .groq_backend import smart_groq_query
            return smart_groq_query(system_message, user_message, func_spec, **model_kwargs)
        except ImportError as e:
            logger.warning(f"Groq backend not available: {e}")
            # Fallback to original backend
            from ai_scientist.treesearch.backend import query as original_query
            return original_query(system_message, user_message, func_spec, **model_kwargs)

    elif any(provider in model for provider in ["anthropic", "claude"]):
        # Use native Anthropic tool calling
        logger.debug(f"Using native Anthropic tool calling for model: {model}")
        output, _, _, _, _ = native_anthropic_query(system_message, user_message, func_spec, **model_kwargs)
        return output

    elif any(provider in model for provider in ["gpt", "openai"]):
        # Use original OpenAI implementation
        logger.debug(f"Using OpenAI function calling for model: {model}")
        from ai_scientist.treesearch.backend.backend_openai import query as openai_query
        output, _, _, _, _ = openai_query(system_message, user_message, func_spec, **model_kwargs)
        return output

    else:
        # Unknown model - try original backend first, fallback on function calling error
        logger.debug(f"Unknown model provider: {model}, trying original backend first")
        try:
            from ai_scientist.treesearch.backend import query as original_query
            # Remove model from model_kwargs to avoid duplicate argument
            kwargs_without_model = {k: v for k, v in model_kwargs.items() if k != "model"}
            return original_query(system_message, user_message, model, func_spec=func_spec, **kwargs_without_model)
        except NotImplementedError as e:
            if "function calling" in str(e).lower() and func_spec is not None:
                logger.info(f"Function calling not supported for {model}, falling back to Anthropic-style tool calling")
                output, _, _, _, _ = native_anthropic_query(system_message, user_message, func_spec, **model_kwargs)
                return output
            else:
                raise


def get_backend_status() -> Dict[str, Any]:
    """Get status of available backends and their capabilities."""
    import os

    status = {
        "groq": {
            "available": bool(os.getenv("GROQ_API_KEY")),
            "supports_functions": True,
            "method": "native_function_calling",
            "advantages": ["Fast inference", "Cost effective", "Open source models"]
        },
        "anthropic": {
            "available": bool(os.getenv("ANTHROPIC_API_KEY")),
            "supports_tools": True,
            "method": "native_tool_use"
        },
        "openai": {
            "available": bool(os.getenv("OPENAI_API_KEY")),
            "supports_functions": True,
            "method": "native_function_calling"
        }
    }

    available_backends = [name for name, info in status.items() if info["available"]]

    return {
        "backends": status,
        "available": available_backends,
        "total_available": len(available_backends),
        "function_calling_support": len([name for name, info in status.items()
                                       if info.get("available") and
                                       (info.get("supports_tools") or info.get("supports_functions"))]),
        "recommended_default": "groq" if status["groq"]["available"] else "anthropic" if status["anthropic"]["available"] else "openai"
    }


# Convenience function for testing
def test_backend_with_simple_function():
    """Test the backend with a simple function spec."""

    # Simple test function spec
    test_spec = FunctionSpec(
        name="test_function",
        description="A simple test function that returns a greeting",
        json_schema={
            "type": "object",
            "properties": {
                "greeting": {
                    "type": "string",
                    "description": "A greeting message"
                },
                "name": {
                    "type": "string",
                    "description": "Name to greet"
                }
            },
            "required": ["greeting", "name"]
        }
    )

    try:
        result = smart_query(
            system_message="You are a helpful assistant",
            user_message="Please greet 'Alice' with a friendly message",
            func_spec=test_spec,
            model="claude-3-5-sonnet-20241022",
            temperature=0.7
        )

        print("Backend test successful!")
        print(f"Result: {result}")
        return True

    except Exception as e:
        print(f"Backend test failed: {e}")
        return False


if __name__ == "__main__":
    # Quick test if run directly
    status = get_backend_status()
    print("Backend Status:")
    print(json.dumps(status, indent=2))

    if status["function_calling_support"] > 0:
        print("\nTesting backend...")
        test_backend_with_simple_function()
    else:
        print("\nNo backends with function calling support available")