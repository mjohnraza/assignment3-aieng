"""LLM client integration using Ollama for Wandering Kettle Café Assistant."""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import ollama

from wandering_kettle_assistant.validation import ToolCallDecision


def get_available_model(preferred: Optional[str] = None) -> str:
    """Detect and return the best available model from Ollama."""
    env_model = os.getenv("OLLAMA_MODEL")
    if env_model:
        return env_model
    if preferred:
        try:
            models_info = ollama.list()
            model_names = [m.model for m in getattr(models_info, "models", [])]
            if preferred in model_names or any(preferred in name for name in model_names):
                return preferred
        except Exception:
            pass

    try:
        models_info = ollama.list()
        model_names = [m.model for m in getattr(models_info, "models", [])]
        for candidate in ["llama3.2:latest", "llama3.2", "qwen2.5:1.5b", "qwen2.5:latest"]:
            for name in model_names:
                if candidate in name:
                    return name
        if model_names:
            return model_names[0]
    except Exception:
        pass

    return "qwen2.5:1.5b"


def parse_react_response(text: str) -> ToolCallDecision:
    """Parse a ReAct output into a ToolCallDecision."""
    text = text.strip()

    # Check for Final Answer
    final_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if final_match:
        thought_match = re.search(r"Thought:\s*(.*?)(?=Final Answer:)", text, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else None
        return ToolCallDecision(
            tool_name="",
            arguments={},
            thought=thought,
            is_final_answer=True,
            final_answer=final_match.group(1).strip(),
        )

    # Extract Thought
    thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", text, re.DOTALL | re.IGNORECASE)
    thought = thought_match.group(1).strip() if thought_match else None

    # Extract Action
    action_match = re.search(r"Action:\s*([a-zA-Z0-9_\-\.]+)", text, re.IGNORECASE)
    tool_name = action_match.group(1).strip() if action_match else ""

    # Extract Action Input
    arguments: Dict[str, Any] = {}
    input_match = re.search(r"Action Input:\s*(\{.*?\}|\[.*?\]|\"[^\"]*\")", text, re.DOTALL | re.IGNORECASE)
    if input_match:
        raw_input = input_match.group(1).strip()
        try:
            parsed = json.loads(raw_input)
            if isinstance(parsed, dict):
                arguments = parsed
            elif isinstance(parsed, str):
                arguments = {"query": parsed}
        except Exception:
            # Fallback if raw text wasn't strict JSON
            arguments = {"query": raw_input.strip('"')}
    else:
        # Check if argument is on the remainder of the line
        line_match = re.search(r"Action Input:\s*(.+)", text, re.IGNORECASE)
        if line_match:
            candidate = line_match.group(1).strip()
            try:
                arguments = json.loads(candidate)
            except Exception:
                arguments = {"query": candidate.strip('"')}

    if not tool_name and not final_match:
        # If model just returned plain text, treat as final answer
        return ToolCallDecision(
            tool_name="",
            arguments={},
            thought=thought,
            is_final_answer=True,
            final_answer=text,
        )

    return ToolCallDecision(
        tool_name=tool_name,
        arguments=arguments,
        thought=thought,
        is_final_answer=False,
    )
