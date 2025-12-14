"""
Groq API Backend for Experimental Setup Evaluation

Provides fast and cost-effective LLM inference using Groq's optimized infrastructure.
Supports Llama models with function calling capabilities.
"""

import json
import time
import logging
from typing import Tuple, Dict, Any, Optional

from ai_scientist.treesearch.backend.utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
import groq

logger = logging.getLogger("ai-scientist")


def groq_query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> Tuple[OutputType, float, int, int, Dict]:
    """
    Groq API query with function calling support.

    Args:
        system_message: System message
        user_message: User message
        func_spec: Function specification for tool calling
        **model_kwargs: Model parameters

    Returns:
        Tuple of (output, request_time, input_tokens, output_tokens, info)
    """
    model_name = model_kwargs.get("model", "llama-3.1-8b-instant")

    # Initialize Groq client
    client = groq.Groq()  # Uses GROQ_API_KEY environment variable

    filtered_kwargs: Dict = {k: v for k, v in model_kwargs.items() if v is not None}

    # Set default max_tokens if not specified
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 8192

    # Handle system message
    messages = []
    if system_message is not None:
        messages.append({"role": "system", "content": system_message})
    if user_message is not None:
        messages.append({"role": "user", "content": user_message})

    # Add function calling if specified
    if func_spec is not None:
        filtered_kwargs["tools"] = [{
            "type": "function",
            "function": {
                "name": func_spec.name,
                "description": func_spec.description,
                "parameters": func_spec.json_schema
            }
        }]
        filtered_kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": func_spec.name}
        }

    # Remove model from filtered_kwargs for the API call
    api_kwargs = {k: v for k, v in filtered_kwargs.items() if k != "model"}

    t0 = time.time()
    try:
        logger.info(f"Sending Groq request... Model: {model_name}")
        response = backoff_create(
            client.chat.completions.create,
            (Exception,),  # Groq client exceptions
            model=model_name,
            messages=messages,
            **api_kwargs,
        )
        logger.info(f"Groq request successful. Duration: {time.time() - t0:.2f}s")
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        raise

    req_time = time.time() - t0

    # Parse response
    if func_spec is not None and response.choices[0].message.tool_calls:
        # Extract function call result
        tool_call = response.choices[0].message.tool_calls[0]
        try:
            output = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse function arguments: {tool_call.function.arguments}")
            raise json.JSONDecodeError(f"Invalid function call response: {e}", tool_call.function.arguments, 0)
    else:
        # Regular text response
        output = response.choices[0].message.content

    # Get token usage (Groq provides this in the response)
    usage = response.usage
    in_tokens = usage.prompt_tokens if usage else 0
    out_tokens = usage.completion_tokens if usage else 0

    info = {
        "stop_reason": response.choices[0].finish_reason,
        "function_call_method": "groq_native" if func_spec else "text",
        "model_used": model_name
    }

    return output, req_time, in_tokens, out_tokens, info


def smart_groq_query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    Smart Groq query that returns just the output.
    Compatible with the unified evaluation backend interface.

    Args:
        system_message: System message
        user_message: User message
        func_spec: Function specification for tool calling
        **model_kwargs: Model parameters

    Returns:
        OutputType: Just the output (string or dict), not the full tuple
    """
    output, _, _, _, _ = groq_query(system_message, user_message, func_spec, **model_kwargs)
    return output


def get_groq_status() -> Dict[str, Any]:
    """Get Groq backend status and capabilities."""
    import os

    has_key = bool(os.getenv("GROQ_API_KEY"))

    return {
        "groq": {
            "available": has_key,
            "supports_functions": True,
            "method": "native_function_calling",
            "models": [
                "llama-3.1-8b-instant",
                "llama-3.1-70b-versatile",
                "llama-3.2-1b-preview",
                "llama-3.2-3b-preview",
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ],
            "advantages": ["Fast inference", "Cost effective", "Open source models"]
        }
    }


# Test function
def test_groq_backend():
    """Test Groq backend with a simple function call."""
    import os

    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY not set - cannot test")
        return False

    test_spec = FunctionSpec(
        name="test_function",
        description="A simple test function",
        json_schema={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "A test message"
                },
                "number": {
                    "type": "integer",
                    "description": "A test number"
                }
            },
            "required": ["message", "number"]
        }
    )

    try:
        result = smart_groq_query(
            system_message="You are a helpful assistant.",
            user_message="Please call the test function with message 'Hello Groq' and number 42",
            func_spec=test_spec,
            model="llama-3.1-8b-instant",
            temperature=0.1
        )

        print("Groq backend test successful!")
        print(f"Result: {result}")
        return True

    except Exception as e:
        print(f"Groq backend test failed: {e}")
        return False


if __name__ == "__main__":
    # Quick test if run directly
    status = get_groq_status()
    print("Groq Backend Status:")
    print(json.dumps(status, indent=2))

    if status["groq"]["available"]:
        print("\nTesting backend...")
        test_groq_backend()
    else:
        print("\nGROQ_API_KEY not available - set it to test the backend")
        print("export GROQ_API_KEY='your-groq-key'")