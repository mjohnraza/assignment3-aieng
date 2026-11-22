"""Iterative ReAct Agent for Wandering Kettle Café Assistant.

Connects to the FastMCP Server via the MCP ClientSession protocol:
User Question
  -> Ollama LLM (Tool Selection & Thought)
  -> Application Tool Executor (Validation)
  -> MCP Client (ClientSession.call_tool())
  -> MCP Server (ChromaDB Vector Retrieval & Policies)
  -> Tool Observation with tool_call_id -> LLM -> Next Decision / Final Answer.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import ollama

from wandering_kettle_assistant.citations import Citation
from wandering_kettle_assistant.llm_client import get_available_model
from wandering_kettle_assistant.mcp_client import MCPClientManager
from wandering_kettle_assistant.validation import (
    RejectionResult,
    ToolCallDecision,
    ToolObservation,
    validate_and_execute_tool,
    validate_tool_arguments,
)

# Tool Schemas for Ollama function calling matching the MCP Server tools
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "lookup_cafe_policy",
            "description": "Lookup official café policies including pet rules (dogs, cats, terrace), outdoor seating, Wi-Fi, laptop rules, and hours.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Policy area such as 'pet_policy', 'outdoor_seating', 'terrace', 'wifi', 'hours'",
                    }
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_menu",
            "description": "Search café food, sandwiches, pastries, and drink items with allergen notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Menu category: 'food', 'pastry', 'drinks', or 'all'",
                        "default": "all",
                    },
                    "include_allergens": {
                        "type": "boolean",
                        "description": "Whether to query allergen notes",
                        "default": True,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "General semantic search of the café knowledge base for menu items, allergens, brewing, hours, seating policies, Wi-Fi, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query terms covering café documentation",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum records to retrieve (1-20)",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
]

AVAILABLE_TOOL_NAMES = ["lookup_cafe_policy", "search_menu", "search"]


@dataclass
class QueryConstraint:
    """A specific evidence requirement that must be satisfied by a SINGLE observation."""
    facet_key: str
    description: str
    # Each checker receives the lowercase text of a SINGLE observation
    checker: Callable[[str], bool]


def extract_query_facets(question: str) -> Dict[str, str]:
    """Identify required informational facets from the user query."""
    q = question.lower()
    facets = {}
    if any(k in q for k in ["pet", "cat", "dog", "animal", "terrace", "sit outside", "outside", "patio"]):
        facets["pet_policy"] = "Outdoor seating and pet/animal policy"
    if any(k in q for k in ["food", "menu", "eat", "sandwich", "pastry", "breakfast", "lunch", "dish"]):
        facets["food_menu"] = "Food menu options"
    if any(k in q for k in ["allerg", "gluten", "wheat", "vegan", "plant", "dairy", "nut", "diet"]):
        facets["allergens_diet"] = "Allergens and dietary accommodations"
    if any(k in q for k in ["hour", "open", "close", "schedule", "when are you"]):
        facets["hours"] = "Operating hours and access"
    if any(k in q for k in ["wifi", "wi-fi", "internet", "laptop", "socket", "outlet", "work"]):
        facets["work_wifi"] = "Laptop and Wi-Fi policy"
    return facets


def _extract_constraints(
    question: str,
    facets: Optional[Dict[str, str]] = None,
) -> List[QueryConstraint]:
    """Extract query-specific constraints from the question text.

    Each constraint specifies the EXACT entity or detail the user asked about,
    not just a broad facet category. The checker verifies that a SINGLE observation
    contains the needed evidence (not across multiple observations combined).
    """
    q = question.lower()
    constraints: List[QueryConstraint] = []

    # 1. Animal / Pet policy
    if "cat" in q:
        constraints.append(QueryConstraint(
            facet_key="pet_policy",
            description="Cat policy on the terrace/outdoors (cats specifically)",
            checker=lambda t: (
                "cat" in t
                and ("terrace" in t or "outdoor" in t or "welcome" in t or "not mentioned" in t or "prohibited" in t)
            ),
        ))
    elif "dog" in q:
        constraints.append(QueryConstraint(
            facet_key="pet_policy",
            description="Dog policy on the terrace/outdoors",
            checker=lambda t: (
                "dog" in t
                and ("terrace" in t or "outdoor" in t or "stairs" in t or "welcome" in t)
            ),
        ))
    elif any(k in q for k in ["pet", "animal", "terrace", "sit outside", "outside", "patio"]):
        constraints.append(QueryConstraint(
            facet_key="pet_policy",
            description="Pet/animal policy on the terrace/outdoors",
            checker=lambda t: (
                ("dog" in t or "animal" in t or "pet" in t)
                and ("terrace" in t or "outdoor" in t or "stairs" in t)
            ),
        ))

    # 2. Food menu
    if any(k in q for k in ["food", "menu", "eat", "sandwich", "pastry", "breakfast", "lunch", "dish"]):
        constraints.append(QueryConstraint(
            facet_key="food_menu",
            description="Food menu options",
            checker=lambda t: any(
                item in t for item in ["sandwich", "baguette", "porridge", "toast", "soup", "hummus", "egg salad", "scrambled"]
            ),
        ))

    # 3. Dietary & Allergens (strictly differentiated)
    if "gluten" in q or "wheat" in q:
        constraints.append(QueryConstraint(
            facet_key="allergens_diet",
            description="Gluten/wheat dietary accommodations",
            checker=lambda t: (
                ("gluten" in t or "wheat" in t or "without wheat" in t)
                and ("bread" in t or "porridge" in t or "soup" in t or "pastry" in t or "pastries" in t)
            ),
        ))
    if "vegan" in q or "plant" in q:
        constraints.append(QueryConstraint(
            facet_key="allergens_diet",
            description="Vegan/plant-based dietary options",
            checker=lambda t: (
                ("plant based" in t or "vegan" in t or "without animal products" in t or "without animal" in t or "without dairy" in t or "no dairy" in t)
            ),
        ))
    if "nut" in q:
        constraints.append(QueryConstraint(
            facet_key="allergens_diet",
            description="Nut allergy information",
            checker=lambda t: ("nut" in t or "almond" in t or "peanut" in t),
        ))
    if any(k in q for k in ["allerg", "dairy", "diet"]) and not ("gluten" in q or "wheat" in q or "vegan" in q or "plant" in q or "nut" in q):
        constraints.append(QueryConstraint(
            facet_key="allergens_diet",
            description="Allergens and dietary accommodations",
            checker=lambda t: any(
                term in t for term in ["gluten", "wheat", "dairy", "plant based", "peanut", "nut", "animal product", "allergen"]
            ),
        ))

    # 4. Operating hours
    if "saturday" in q and "sunday" in q:
        constraints.append(QueryConstraint(
            facet_key="hours",
            description="Saturday operating hours (9:00 to 22:00)",
            checker=lambda t: "saturday" in t and "9:00" in t and "22:00" in t,
        ))
        constraints.append(QueryConstraint(
            facet_key="hours",
            description="Sunday operating hours (9:00 to 17:00)",
            checker=lambda t: "sunday" in t and "9:00" in t and "17:00" in t,
        ))
    elif "saturday" in q:
        constraints.append(QueryConstraint(
            facet_key="hours",
            description="Saturday operating hours (9:00 to 22:00)",
            checker=lambda t: "saturday" in t and "9:00" in t and "22:00" in t,
        ))
    elif "sunday" in q:
        constraints.append(QueryConstraint(
            facet_key="hours",
            description="Sunday operating hours (9:00 to 17:00)",
            checker=lambda t: "sunday" in t and "9:00" in t and "17:00" in t,
        ))
    elif any(k in q for k in ["hour", "open", "close", "schedule", "when are you"]):
        constraints.append(QueryConstraint(
            facet_key="hours",
            description="Operating hours and access",
            checker=lambda t: (
                any(d in t for d in ["monday", "friday", "saturday", "sunday", "schedule"])
                and any(ts in t for ts in ["8:00", "9:00", "18:00", "22:00", "17:00", "open", "closed"])
            ),
        ))

    # 5. Working environment: Laptop policy, Wi-Fi credentials, Power sockets
    if "laptop" in q or "stay and work" in q:
        constraints.append(QueryConstraint(
            facet_key="work_wifi",
            description="Laptop policy (curfew after 6pm)",
            checker=lambda t: (
                ("laptop" in t or "stay and work" in t)
                and ("friday" in t or "saturday" in t or "conversation" in t or "ninety minutes" in t)
            ),
        ))
    if any(k in q for k in ["wifi", "wi-fi", "internet", "wireless"]):
        constraints.append(QueryConstraint(
            facet_key="work_wifi",
            description="Wi-Fi credentials (kettle-guest)",
            checker=lambda t: "kettle-guest" in t and ("password" in t or "coaster" in t or "wifi" in t or "wireless" in t),
        ))
    if any(k in q for k in ["socket", "outlet", "power"]):
        constraints.append(QueryConstraint(
            facet_key="work_wifi",
            description="Power socket locations (window bench/bookshelf)",
            checker=lambda t: (
                ("socket" in t or "outlet" in t or "power" in t)
                and ("window bench" in t or "book" in t or "six" in t or "two" in t)
            ),
        ))
    if "work" in q and not ("laptop" in q or "stay and work" in q or any(k in q for k in ["wifi", "wi-fi", "internet", "socket", "outlet"])):
        constraints.append(QueryConstraint(
            facet_key="work_wifi",
            description="Laptop and Wi-Fi policy",
            checker=lambda t: (
                ("kettle-guest" in t or "socket" in t or "window bench" in t)
                and ("wifi" in t or "laptop" in t or "work" in t)
            ),
        ))

    return constraints


def check_evidence_coverage(
    facets: Dict[str, str],
    trace: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Determine which facets have retrieved evidence vs which remain unverified.

    Evidence evaluation rules:
    1. Ignore failed, timed-out, and malformed tool calls (is_error=True or execution_status != 'success').
    2. Never count tool names or input arguments as retrieved evidence.
    3. Check each constraint against each observation INDEPENDENTLY (not against combined corpus).
    4. Only a SINGLE successful observation must satisfy a constraint on its own.
    5. Mark unsupported facets as missing.
    """
    if not facets:
        return [], []

    covered: List[str] = []
    missing: List[str] = []

    # 1. Filter strictly for successful observations without errors
    successful_contents: List[str] = []
    for t in trace:
        # Check error status
        is_err = t.get("is_error", False)
        status = t.get("execution_status", t.get("observation_status", "error"))
        if is_err or status != "success":
            continue

        # Extract actual observation content (never arguments or tool names)
        obs = t.get("observation")
        if isinstance(obs, dict) and "content" in obs:
            content = obs["content"]
        else:
            content = t.get("result", t.get("observation_preview", ""))

        if isinstance(content, str) and content.strip():
            successful_contents.append(content.lower())

    # 2. Build query-specific constraints from the question text
    # We need the original question — reconstruct from facets or use a broad check
    # For backward compatibility, we check facets directly when no constraints available
    # _extract_constraints is called by the agent with the original question

    # 2. Check each facet against individual successful observations
    for facet_key, facet_desc in facets.items():
        is_facet_covered = False

        for obs_text in successful_contents:
            if facet_key == "pet_policy":
                if (
                    ("dog" in obs_text or "animal" in obs_text)
                    and ("terrace" in obs_text or "outdoor" in obs_text or "stairs" in obs_text)
                ):
                    is_facet_covered = True
                    break
            elif facet_key == "food_menu":
                if any(
                    item in obs_text
                    for item in ["sandwich", "baguette", "porridge", "toast", "soup", "hummus", "egg salad"]
                ):
                    is_facet_covered = True
                    break
            elif facet_key == "allergens_diet":
                if any(
                    term in obs_text
                    for term in ["gluten", "wheat", "dairy", "plant based", "peanut", "nut", "animal product", "without wheat"]
                ):
                    is_facet_covered = True
                    break
            elif facet_key == "hours":
                if any(
                    day in obs_text for day in ["monday", "friday", "saturday", "sunday", "schedule"]
                ) and any(
                    time_str in obs_text for time_str in ["8:00", "9:00", "18:00", "22:00", "17:00", "open", "closed"]
                ):
                    is_facet_covered = True
                    break
            elif facet_key == "work_wifi":
                if (
                    "kettle-guest" in obs_text or "socket" in obs_text or "window bench" in obs_text
                ) and (
                    "wifi" in obs_text or "laptop" in obs_text or "work" in obs_text
                ):
                    is_facet_covered = True
                    break

        if is_facet_covered:
            covered.append(facet_desc)
        else:
            missing.append(facet_desc)

    return covered, missing


def check_evidence_coverage_with_constraints(
    constraints: List['QueryConstraint'],
    trace: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Check evidence coverage using query-specific constraints.

    Each constraint must be satisfied by at least one SINGLE successful observation
    independently (not by combining keywords across multiple observations).
    """
    if not constraints:
        return [], []

    covered: List[str] = []
    missing: List[str] = []

    # Collect successful observation contents
    successful_contents: List[str] = []
    for t in trace:
        is_err = t.get("is_error", False)
        status = t.get("execution_status", t.get("observation_status", "error"))
        if is_err or status != "success":
            continue

        obs = t.get("observation")
        if isinstance(obs, dict) and "content" in obs:
            content = obs["content"]
        else:
            content = t.get("result", t.get("observation_preview", ""))

        if isinstance(content, str) and content.strip():
            successful_contents.append(content.lower())

    # Check each constraint independently against each observation
    for constraint in constraints:
        satisfied = False
        for obs_text in successful_contents:
            if constraint.checker(obs_text):
                satisfied = True
                break
        if satisfied:
            covered.append(constraint.description)
        else:
            missing.append(constraint.description)

    return covered, missing


SYSTEM_PROMPT = """You are a helpful and honest knowledge assistant for "The Wandering Kettle" café.
Answer user requests by reasoning through what information is needed, calling external tools iteratively via the Model Context Protocol (MCP), and stopping only when you have gathered sufficient evidence.

CRITICAL GUIDELINES:
1. Reason before acting: State what information is needed and which tool to query next.
2. Multi-step queries: If the user asks about multiple aspects (e.g. pet policy, food menu, AND allergies), inspect each aspect with a corresponding tool call before providing the final answer.
3. Grounding & Citations: Ground your response strictly on retrieved evidence. Quote policies and menu items directly from observations.
4. Distinguish confirmed from unconfirmed/missing: If a detail (like cats specifically) is not mentioned in the policy (e.g. if the policy only mentions dogs and assistance animals on the terrace), explicitly highlight that distinction rather than assuming or inventing.
5. Final Answer: Output a comprehensive, structured response once all information is verified.
"""


class ReActAgent:
    """
    Production Iterative ReAct Agent:
    Connects to the MCP server through an MCP ClientSession.

    Loop structure:
    while steps < max_steps:
      1. ask_llm(messages, available_tools) -> decision
      2. if decision is final_answer -> return
      3. validate arguments against schema
      4. execute_selected_mcp_tool(decision) via ClientSession.call_tool()
      5. messages += [tool_call, tool_result_with_id]
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_steps: int = 6,
        available_tools: Optional[List[str]] = None,
        mcp_client: Optional[MCPClientManager] = None,
    ):
        self.model_name = model_name or get_available_model()
        self.max_steps = max_steps
        self.available_tools = available_tools or AVAILABLE_TOOL_NAMES
        self.mcp_client = mcp_client
        # Duplicate-call detection: tracks (tool_name, frozen_args) already executed
        self._executed_calls: set = set()

    async def arun(
        self,
        user_question: str,
        mcp_client: Optional[MCPClientManager] = None,
    ) -> Dict[str, Any]:
        """Execute the asynchronous ReAct loop connecting to the MCP Server."""
        client = mcp_client or self.mcp_client
        created_client = False

        if client is None:
            client = MCPClientManager()
            await client.connect()
            created_client = True

        try:
            self._executed_calls = set()  # Reset per-run
            steps = 0
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_question},
            ]
            trace: List[Dict[str, Any]] = []

            print(f"\n[Starting ReAct Agent with Model: {self.model_name}]")
            print(f"Goal: {user_question}\n")

            while steps < self.max_steps:
                steps += 1
                print(f"{'=' * 30} ITERATION {steps} {'=' * 30}")

                # Step 1: Query LLM with exposed tool declarations
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    tools=TOOLS_SCHEMA,
                    options={"temperature": 0.1},
                )
                msg = response["message"]
                thought_text = msg.get("content", "").strip()
                tool_calls = msg.get("tool_calls")

                print(f"[Reasoning/Thought]:\n{thought_text if thought_text else '(Direct tool selection)'}\n")

                # Stopping Condition: If no tool calls were requested, LLM attempted to conclude
                if not tool_calls:
                    query_constraints = _extract_constraints(user_question)
                    covered, missing = check_evidence_coverage_with_constraints(query_constraints, trace)

                    # For multi-part queries, if some aspects lack evidence, prevent premature stopping
                    if len(query_constraints) > 1 and missing and steps < self.max_steps:
                        print(f">> [Evidence Completeness Interceptor]: Missing evidence for: {missing}. Forcing continuation.")
                        messages.append(msg)
                        messages.append({
                            "role": "user",
                            "content": (
                                f"[Evidence Completeness Guard]: You attempted to conclude, but you have not yet gathered documentation for: {', '.join(missing)}. "
                                "Please invoke the appropriate tool (e.g. lookup_cafe_policy, search_menu, or search) before providing your final answer."
                            ),
                        })
                        continue

                    print(">> [Decision]: Sufficient evidence gathered. Returning Final Answer.\n")
                    final_output = thought_text
                    if missing:
                        final_output += (
                            f"\n\n[Deterministic Safeguard Notice: Verified evidence gathered for: {', '.join(covered) if covered else 'None'}. "
                            f"Unverified aspects: {', '.join(missing)}. Please verify unconfirmed details directly with café staff.]"
                        )

                    trace.append({
                        "step": steps,
                        "thought": thought_text,
                        "action": "final_answer",
                        "final_answer": final_output,
                    })
                    return {
                        "question": user_question,
                        "status": "completed",
                        "iterations": steps,
                        "trace": trace,
                        "final_answer": final_output,
                        "covered_facets": covered,
                        "missing_facets": missing,
                    }

                # LLM selected tool calls
                messages.append(msg)

                for call_idx, call in enumerate(tool_calls, start=1):
                    func = call["function"] if isinstance(call, dict) else getattr(call, "function", call)
                    tool_name = func["name"] if isinstance(func, dict) else getattr(func, "name", str(func))
                    raw_args = func["arguments"] if isinstance(func, dict) else getattr(func, "arguments", {})
                    call_id = getattr(call, "id", None) or (call.get("id") if isinstance(call, dict) else f"call_{steps}_{call_idx:02d}")

                    if isinstance(raw_args, str):
                        try:
                            arguments = json.loads(raw_args)
                        except Exception:
                            arguments = {"query": raw_args}
                    elif not isinstance(raw_args, dict):
                        # Guard against malformed arguments (None, list, int, etc.)
                        arguments = {"query": str(raw_args)} if raw_args is not None else {}
                    else:
                        arguments = raw_args

                    # Normalize alias names if needed
                    if tool_name == "search_cafe_knowledge_base":
                        tool_name = "search"

                    # Duplicate-call detection: skip identical (tool, args) pairs
                    call_fingerprint = (tool_name, json.dumps(arguments, sort_keys=True))
                    if call_fingerprint in self._executed_calls:
                        print(f" -> [Duplicate Detection]: Identical call to '{tool_name}' with same arguments already executed. Skipping.")
                        messages.append({
                            "role": "tool",
                            "name": tool_name,
                            "tool_call_id": call_id if isinstance(call_id, str) else f"call_{steps}_{call_idx:02d}",
                            "content": f"Duplicate call skipped: '{tool_name}' was already invoked with these arguments. Use the previous observation.",
                        })
                        trace.append({
                            "step": steps,
                            "call_id": call_id,
                            "thought": thought_text,
                            "action": tool_name,
                            "arguments": arguments,
                            "observation_status": "duplicate_skipped",
                            "observation_preview": "Duplicate call skipped.",
                        })
                        continue

                    decision = ToolCallDecision(
                        tool_name=tool_name,
                        arguments=arguments,
                        thought=thought_text,
                        call_id=call_id,
                    )

                    print(f"[Action]: {tool_name}")
                    print(f"[Action Input]: {json.dumps(arguments, indent=2)}")

                    # Step 2: Validate against schema
                    val_result = validate_tool_arguments(tool_name, arguments)

                    if isinstance(val_result, RejectionResult):
                        obs_text = f"Error: Validation failed: {val_result.error_message()}"
                        observation = ToolObservation(
                            tool_name=tool_name,
                            status="validation_error",
                            result=obs_text,
                            is_error=True,
                            call_id=call_id,
                        )
                    elif tool_name not in self.available_tools:
                        obs_text = f"Error: Unknown tool '{tool_name}'. Available: {self.available_tools}"
                        observation = ToolObservation(
                            tool_name=tool_name,
                            status="error",
                            result=obs_text,
                            is_error=True,
                            call_id=call_id,
                        )
                    else:
                        # Step 3: Execute across MCP Client Transport
                        clean_args = val_result
                        print(f" -> Invoking MCP Server tool '{tool_name}' via ClientSession.call_tool()...")
                        try:
                            mcp_result = await client.call_tool(tool_name, clean_args)
                            is_tool_error = getattr(mcp_result, "is_error", False)
                            content_str = str(mcp_result)
                            observation = ToolObservation(
                                tool_name=tool_name,
                                status="error" if is_tool_error else "success",
                                result=content_str,
                                is_error=is_tool_error,
                                call_id=call_id,
                            )
                        except Exception as exc:
                            observation = ToolObservation(
                                tool_name=tool_name,
                                status="error",
                                result=f"MCP invocation error: {exc}",
                                is_error=True,
                                call_id=call_id,
                            )

                    print(f"[Observation - Status: {observation.status}]:")
                    preview = str(observation.result).strip().splitlines()
                    for line in preview[:5]:
                        print(f"    {line}")
                    if len(preview) > 5:
                        print(f"    ... ({len(preview) - 5} lines truncated)")
                    print()

                    # Step 4: Append tool result with explicit name and tool_call_id
                    messages.append({
                        "role": "tool",
                        "name": tool_name,
                        "tool_call_id": call_id,
                        "content": str(observation.result),
                    })

                    # Record this call in the duplicate-detection set
                    self._executed_calls.add(call_fingerprint)

                    trace.append({
                        "step": steps,
                        "call_id": call_id,
                        "thought": thought_text,
                        "action": tool_name,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "execution_status": observation.status,
                        "is_error": observation.is_error,
                        "observation_status": observation.status,
                        "observation": {
                            "content": str(observation.result),
                        },
                        "observation_preview": str(observation.result)[:250],
                    })

            # Safeguard: Max steps reached with Deterministic Safe-Stop Mechanism
            print(">> [Safeguard]: Max iterations reached. Executing deterministic safe-stop synthesis.\n")
            query_constraints = _extract_constraints(user_question)
            covered, missing = check_evidence_coverage_with_constraints(query_constraints, trace)

            messages.append({
                "role": "user",
                "content": "You have reached the maximum step limit. Synthesize your final answer now based solely on the retrieved evidence above. Distinguish confirmed from missing facts.",
            })
            final_call = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": 0.1},
            )
            final_answer = final_call["message"]["content"].strip()

            # Deterministic safe-stop mechanism: append safety notice if evidence was incomplete
            if missing:
                safe_notice = (
                    f"\n\n[Deterministic Safeguard Notice: The assistant gathered verified documentation for: {', '.join(covered) if covered else 'None'}. "
                    f"Evidence was NOT retrieved for: {', '.join(missing)}. Answers for unverified aspects must be confirmed with café staff directly.]"
                )
                final_answer += safe_notice

            return {
                "question": user_question,
                "status": "max_steps_reached",
                "iterations": steps,
                "trace": trace,
                "final_answer": final_answer,
                "covered_facets": covered,
                "missing_facets": missing,
            }

        finally:
            if created_client:
                await client.close()

    def run(self, user_question: str) -> Dict[str, Any]:
        """Synchronous runner for the ReAct Agent."""
        return asyncio.run(self.arun(user_question))
