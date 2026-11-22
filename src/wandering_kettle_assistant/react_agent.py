"""Iterative ReAct Agent for Wandering Kettle Café Assistant."""

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import ollama

from wandering_kettle_assistant.citations import Citation
from wandering_kettle_assistant.llm_client import get_available_model
from wandering_kettle_assistant.retrieval import (
    lookup_policy,
    search_knowledge_base,
    search_menu,
)
from wandering_kettle_assistant.validation import (
    RejectionResult,
    ToolCallDecision,
    ToolObservation,
    validate_and_execute_tool,
    validate_search_args,
)

# Tool Schemas for Ollama function calling
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
            "description": "Search café food, sandwiches, pastries, and drink items.",
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
            "description": "General search of the café knowledge base for menu items, allergens, brewing, hours, seating policies, Wi-Fi, etc.",
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

SYSTEM_PROMPT = """You are a helpful and honest knowledge assistant for "The Wandering Kettle" café.
Answer user requests by reasoning through what information is needed, calling external tools iteratively, and stopping only when you have gathered sufficient evidence.

CRITICAL GUIDELINES:
1. Reason before acting: State what information is needed and which tool to query next.
2. Multi-step queries: If the user asks about multiple aspects (e.g. pet policy, food menu, AND allergies), inspect each aspect with a corresponding tool call before providing the final answer.
3. Grounding & Citations: Ground your response strictly on retrieved evidence. Quote policies and menu items directly from observations.
4. Distinguish confirmed from unconfirmed/missing: If a detail (like cats specifically) is not mentioned in the policy (e.g. if the policy only mentions dogs and assistance animals), explicitly highlight that distinction rather than assuming or inventing.
5. Final Answer: Output a comprehensive, structured response once all information is verified.
"""


def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Dispatches the tool call to the local café knowledge retrieval functions."""
    if name == "search":
        val = validate_search_args(arguments)
        if isinstance(val, RejectionResult):
            errs = "; ".join(e.message for e in val.errors)
            return f"Validation Error: {errs}"
        return search_knowledge_base(query=val.query, top_k=val.top_k)

    if name == "lookup_cafe_policy":
        topic = arguments.get("topic", "")
        if not topic:
            return "Validation Error: 'topic' argument is required for lookup_cafe_policy."
        return lookup_policy(topic=topic)

    if name == "search_menu":
        category = arguments.get("category", "all")
        include_allergens = arguments.get("include_allergens", True)
        return search_menu(category=category, include_allergens=include_allergens)

    raise ValueError(f"Unknown tool: '{name}'")


class ReActAgent:
    """
    Explicit Iterative ReAct Agent:
    Loop: while steps < max_steps:
      1. ask_llm(messages, available_tools) -> decision
      2. if decision is final_answer -> return
      3. execute_selected_mcp_tool(decision) -> observation
      4. messages += [tool_call, tool_result]
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_steps: int = 6,
        available_tools: Optional[List[str]] = None,
    ):
        self.model_name = model_name or get_available_model()
        self.max_steps = max_steps
        self.available_tools = available_tools or AVAILABLE_TOOL_NAMES

    def run(self, user_question: str) -> Dict[str, Any]:
        """Execute the iterative ReAct loop until completion or max_steps reached."""
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

            # Step 1: Ask LLM with tools
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

            # Stopping Condition: If no tool calls were made, LLM decided it has enough evidence
            if not tool_calls:
                print(">> [Decision]: Sufficient evidence gathered. Returning Final Answer.\n")
                trace.append({
                    "step": steps,
                    "thought": thought_text,
                    "action": "final_answer",
                    "final_answer": thought_text,
                })
                return {
                    "question": user_question,
                    "status": "completed",
                    "iterations": steps,
                    "trace": trace,
                    "final_answer": thought_text,
                }

            # LLM decided to take an Action
            messages.append(msg)

            for call in tool_calls:
                func = call["function"] if isinstance(call, dict) else getattr(call, "function", call)
                tool_name = func["name"] if isinstance(func, dict) else getattr(func, "name", str(func))
                raw_args = func["arguments"] if isinstance(func, dict) else getattr(func, "arguments", {})

                if isinstance(raw_args, str):
                    try:
                        arguments = json.loads(raw_args)
                    except Exception:
                        arguments = {"query": raw_args}
                else:
                    arguments = raw_args

                decision = ToolCallDecision(
                    tool_name=tool_name,
                    arguments=arguments,
                    thought=thought_text,
                )

                print(f"[Action]: {tool_name}")
                print(f"[Action Input]: {json.dumps(arguments, indent=2)}")

                # Step 2: Validate and execute tool
                observation: ToolObservation = validate_and_execute_tool(
                    decision=decision,
                    available_tools=self.available_tools,
                    executor_fn=execute_tool,
                )

                print(f"[Observation - Status: {observation.status}]:")
                preview = str(observation.result).strip().splitlines()
                for line in preview[:5]:
                    print(f"    {line}")
                if len(preview) > 5:
                    print(f"    ... ({len(preview) - 5} lines truncated)")
                print()

                # Step 3: Append observation back into conversation history
                messages.append({
                    "role": "tool",
                    "content": str(observation.result),
                })

                trace.append({
                    "step": steps,
                    "thought": thought_text,
                    "action": tool_name,
                    "arguments": arguments,
                    "observation_status": observation.status,
                    "observation_preview": str(observation.result)[:250],
                })

        # Safeguard: Maximum steps reached
        print(">> [Safeguard]: Max iterations reached. Prompting LLM for grounded final synthesis.\n")
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

        return {
            "question": user_question,
            "status": "max_steps_reached",
            "iterations": steps,
            "trace": trace,
            "final_answer": final_answer,
        }
