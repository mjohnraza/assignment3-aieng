"""Argument and tool call validation schemas and logic."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import json


# ---------------------------------------------------------------------------
# Search argument validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidatedSearchArgs:
    query: str
    top_k: int


@dataclass(frozen=True)
class ValidationError:
    field: str
    message: str


@dataclass(frozen=True)
class RejectionResult:
    errors: list[ValidationError]

    @property
    def is_valid(self) -> bool:
        return False


DEFAULT_TOP_K = 5
MIN_TOP_K = 1
MAX_TOP_K = 20


def validate_search_args(
    raw_args: dict[str, Any]
) -> Union[ValidatedSearchArgs, RejectionResult]:
    """Validate and sanitize raw search arguments against expected boundaries."""
    errors: list[ValidationError] = []

    # Query validation: required, must be a non-empty string
    query_val = raw_args.get("query")
    if query_val is None:
        errors.append(ValidationError(field="query", message="Field 'query' is required."))
        clean_query = ""
    elif not isinstance(query_val, str):
        errors.append(
            ValidationError(
                field="query",
                message=f"Field 'query' must be a string, got {type(query_val).__name__}."
            )
        )
        clean_query = ""
    else:
        clean_query = query_val.strip()
        if not clean_query:
            errors.append(
                ValidationError(field="query", message="Field 'query' cannot be empty or whitespace only.")
            )

    # top_k coercion and boundary checks
    parsed_top_k = DEFAULT_TOP_K
    if "top_k" in raw_args and raw_args["top_k"] is not None:
        top_k_raw = raw_args["top_k"]
        try:
            # Handle string-encoded integers from JSON / tool call representations
            if isinstance(top_k_raw, bool):
                raise ValueError("Booleans are not valid integers for top_k.")
            candidate = int(top_k_raw)
            if candidate < MIN_TOP_K or candidate > MAX_TOP_K:
                errors.append(
                    ValidationError(
                        field="top_k",
                        message=f"Field 'top_k' must be between {MIN_TOP_K} and {MAX_TOP_K}, got {candidate}."
                    )
                )
            else:
                parsed_top_k = candidate
        except (ValueError, TypeError):
            errors.append(
                ValidationError(
                    field="top_k",
                    message=f"Field 'top_k' could not be coerced to an integer: {top_k_raw!r}."
                )
            )

    if errors:
        return RejectionResult(errors=errors)

    return ValidatedSearchArgs(query=clean_query, top_k=parsed_top_k)


# ---------------------------------------------------------------------------
# Tool call decision and observation dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ToolCallDecision:
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    thought: Optional[str] = None
    is_final_answer: bool = False
    final_answer: Optional[str] = None


@dataclass
class ToolObservation:
    tool_name: str
    status: str
    result: Any
    is_error: bool = False


def validate_and_execute_tool(
    decision: ToolCallDecision,
    available_tools: List[str],
    executor_fn: Any = None,
) -> ToolObservation:
    """Validate that the requested tool is exposed before executing it."""
    # Short-circuit execution if the model requests an unknown or unexposed tool
    if decision.tool_name not in available_tools:
        error_payload = {
            "error": f"Unknown tool '{decision.tool_name}'.",
            "available_tools": available_tools,
            "hint": "Select only from the available tools registered on the MCP server.",
        }
        return ToolObservation(
            tool_name=decision.tool_name,
            status="error",
            result=json.dumps(error_payload, indent=2),
            is_error=True,
        )

    if executor_fn is None:
        return ToolObservation(
            tool_name=decision.tool_name,
            status="mock_executed",
            result=f"Tool {decision.tool_name} executed with args {decision.arguments}",
            is_error=False,
        )

    try:
        result = executor_fn(decision.tool_name, decision.arguments)
        return ToolObservation(
            tool_name=decision.tool_name,
            status="success",
            result=result,
            is_error=False,
        )
    except Exception as exc:
        return ToolObservation(
            tool_name=decision.tool_name,
            status="error",
            result=f"Execution error: {exc}",
            is_error=True,
        )
