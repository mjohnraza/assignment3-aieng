"""Argument and tool call validation schemas and logic for all café tools."""

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
class ValidatedPolicyArgs:
    topic: str


@dataclass(frozen=True)
class ValidatedMenuArgs:
    category: str
    include_allergens: bool


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

    def error_message(self) -> str:
        return "; ".join(err.message for err in self.errors)


DEFAULT_TOP_K = 5
MIN_TOP_K = 1
MAX_TOP_K = 20
ALLOWED_MENU_CATEGORIES = {"all", "food", "pastry", "drinks", "beverage", "beverages"}


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


def validate_policy_args(
    raw_args: dict[str, Any]
) -> Union[ValidatedPolicyArgs, RejectionResult]:
    """Validate arguments for lookup_cafe_policy."""
    errors: list[ValidationError] = []
    topic_val = raw_args.get("topic")

    if topic_val is None:
        errors.append(ValidationError(field="topic", message="Field 'topic' is required."))
        clean_topic = ""
    elif not isinstance(topic_val, str):
        errors.append(
            ValidationError(
                field="topic",
                message=f"Field 'topic' must be a string, got {type(topic_val).__name__}."
            )
        )
        clean_topic = ""
    else:
        clean_topic = topic_val.strip()
        if not clean_topic:
            errors.append(
                ValidationError(field="topic", message="Field 'topic' cannot be empty or whitespace only.")
            )
        elif len(clean_topic) > 100:
            errors.append(
                ValidationError(field="topic", message="Field 'topic' exceeds maximum length of 100 characters.")
            )

    if errors:
        return RejectionResult(errors=errors)

    return ValidatedPolicyArgs(topic=clean_topic)


def validate_menu_args(
    raw_args: dict[str, Any]
) -> Union[ValidatedMenuArgs, RejectionResult]:
    """Validate arguments for search_menu."""
    errors: list[ValidationError] = []

    cat_val = raw_args.get("category", "all")
    if not isinstance(cat_val, str):
        errors.append(
            ValidationError(
                field="category",
                message=f"Field 'category' must be a string, got {type(cat_val).__name__}."
            )
        )
        clean_category = "all"
    else:
        clean_category = cat_val.strip().lower()
        if clean_category not in ALLOWED_MENU_CATEGORIES:
            errors.append(
                ValidationError(
                    field="category",
                    message=f"Invalid category '{clean_category}'. Allowed: {sorted(ALLOWED_MENU_CATEGORIES)}."
                )
            )

    allergens_val = raw_args.get("include_allergens", True)
    if not isinstance(allergens_val, bool):
        if isinstance(allergens_val, str) and allergens_val.lower() in ("true", "false"):
            allergens_val = allergens_val.lower() == "true"
        else:
            errors.append(
                ValidationError(
                    field="include_allergens",
                    message=f"Field 'include_allergens' must be a boolean, got {type(allergens_val).__name__}."
                )
            )

    if errors:
        return RejectionResult(errors=errors)

    return ValidatedMenuArgs(category=clean_category, include_allergens=bool(allergens_val))


def validate_tool_arguments(
    tool_name: str,
    arguments: Dict[str, Any]
) -> Union[Dict[str, Any], RejectionResult]:
    """Validate arguments for any registered café tool against its exact schema."""
    if tool_name in ("search", "search_cafe_knowledge_base"):
        res = validate_search_args(arguments)
        if isinstance(res, RejectionResult):
            return res
        return {"query": res.query, "top_k": res.top_k}

    if tool_name == "lookup_cafe_policy":
        res = validate_policy_args(arguments)
        if isinstance(res, RejectionResult):
            return res
        return {"topic": res.topic}

    if tool_name == "search_menu":
        res = validate_menu_args(arguments)
        if isinstance(res, RejectionResult):
            return res
        return {"category": res.category, "include_allergens": res.include_allergens}

    return RejectionResult(errors=[ValidationError(field="tool_name", message=f"Unknown tool '{tool_name}'.")])


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
    call_id: Optional[str] = None


@dataclass
class ToolObservation:
    tool_name: str
    status: str
    result: Any
    is_error: bool = False
    call_id: Optional[str] = None


def validate_and_execute_tool(
    decision: ToolCallDecision,
    available_tools: List[str],
    executor_fn: Any = None,
) -> ToolObservation:
    """
    Validate that the requested tool is exposed and arguments match schema before executing.
    Supports synchronous and callable executors.
    """
    call_id = decision.call_id or f"call_{decision.tool_name}"

    # 1. Short-circuit execution if the model requests an unknown or unexposed tool
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
            call_id=call_id,
        )

    # 2. Comprehensive argument validation against the tool's schema
    val_result = validate_tool_arguments(decision.tool_name, decision.arguments)
    if isinstance(val_result, RejectionResult):
        error_payload = {
            "error": "Argument schema validation failed.",
            "tool": decision.tool_name,
            "details": [e.message for e in val_result.errors],
        }
        return ToolObservation(
            tool_name=decision.tool_name,
            status="validation_error",
            result=json.dumps(error_payload, indent=2),
            is_error=True,
            call_id=call_id,
        )

    # Validated arguments
    clean_args = val_result

    if executor_fn is None:
        return ToolObservation(
            tool_name=decision.tool_name,
            status="mock_executed",
            result=f"Tool {decision.tool_name} executed with args {clean_args}",
            is_error=False,
            call_id=call_id,
        )

    try:
        result = executor_fn(decision.tool_name, clean_args)
        return ToolObservation(
            tool_name=decision.tool_name,
            status="success",
            result=result,
            is_error=False,
            call_id=call_id,
        )
    except Exception as exc:
        return ToolObservation(
            tool_name=decision.tool_name,
            status="error",
            result=f"Execution error: {exc}",
            is_error=True,
            call_id=call_id,
        )
