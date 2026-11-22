"""Part C (Steps 12-15): LLM-enabled RAG with MCP Tool Calling and Safeguards.

Full sequence implemented:
User question -> LLM tool decision -> MCP tool invocation (via ClientSession.call_tool())
-> retrieved context -> LLM grounded response.
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List
import ollama

from wandering_kettle_assistant.llm_client import get_available_model
from wandering_kettle_assistant.mcp_client import MCPClientManager
from wandering_kettle_assistant.validation import validate_tool_arguments, RejectionResult

# Tool schemas exposed to the LLM (matching FastMCP tools)
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search Wandering Kettle cafe menu, policies, business hours, and general documentation",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query terms for café documentation",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of relevant document passages to retrieve",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_cafe_policy",
            "description": "Lookup official café policies including pet rules, outdoor seating, Wi-Fi, and hours",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Policy area (e.g., 'pet_policy', 'outdoor_seating', 'wifi', 'hours')",
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
            "description": "Search café food and drink menu items with detailed allergen notes",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Menu category (e.g., 'food', 'drinks', 'pastry', 'all')",
                        "default": "all",
                    },
                    "include_allergens": {
                        "type": "boolean",
                        "description": "Whether to include allergen notes and warnings",
                        "default": True,
                    },
                },
            },
        },
    },
]

SYSTEM_PROMPT = """You are a helpful and honest knowledge assistant for "The Wandering Kettle" café.
You have access to external tools that retrieve official café documentation via the Model Context Protocol (MCP).

STRICT INSTRUCTIONS:
1. Grounding: Answer strictly using facts retrieved from the café documents.
2. Missing Information Safeguard: If the retrieved documents do not contain the answer or indicate information is unavailable, explicitly state that the information is unavailable or unconfirmed. DO NOT make up or hallucinate policies, parking, EV charging, menu items, prices, or amenities.
3. Citations: Quote or reference the source document (e.g. faq.txt, allergens.txt, hours_access.txt).
"""


async def answer_question_with_rag(
    question: str,
    model_name: str,
    mcp_client: MCPClientManager,
) -> Dict[str, Any]:
    """Execute the full MCP-grounded RAG sequence for a user question."""
    print("\n" + "=" * 80)
    print(f"USER QUESTION: {question}")
    print("=" * 80)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    print("[Step 1: LLM Tool Decision]")
    first_response = ollama.chat(
        model=model_name,
        messages=messages,
        tools=TOOLS_SCHEMA,
    )
    first_msg = first_response["message"]

    tool_calls = first_msg.get("tool_calls")
    tool_trace: List[Dict[str, Any]] = []

    # If model didn't trigger structured tool_calls, fallback to search tool
    if not tool_calls:
        print("Model generated direct text. Routing through MCP 'search' tool...")
        tool_calls = [
            {
                "id": "call_fallback_001",
                "function": {
                    "name": "search",
                    "arguments": {"query": question, "top_k": 3},
                }
            }
        ]

    messages.append(first_msg)

    for idx, call in enumerate(tool_calls, start=1):
        func = call["function"] if isinstance(call, dict) else getattr(call, "function", call)
        name = func["name"] if isinstance(func, dict) else getattr(func, "name", str(func))
        raw_args = func["arguments"] if isinstance(func, dict) else getattr(func, "arguments", {})
        call_id = getattr(call, "id", None) or (call.get("id") if isinstance(call, dict) else f"call_{idx:03d}")

        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {"query": raw_args}
        else:
            args = raw_args

        # Map legacy names if any
        if name == "search_cafe_knowledge_base":
            name = "search"

        print(f" -> Selected Tool: '{name}' (Call ID: {call_id})")
        print(f" -> Tool Arguments: {json.dumps(args)}")

        # Validate arguments against tool schema before MCP transmission
        val_check = validate_tool_arguments(name, args)
        if isinstance(val_check, RejectionResult):
            print(f" -> Validation rejected: {val_check.error_message()}")
            tool_result = f"Error: Argument validation failed: {val_check.error_message()}"
        else:
            clean_args = val_check
            print("\n[Step 2: MCP Tool Invocation via ClientSession.call_tool()]")
            print(f" -> Dispatching to MCP Server across Stdio transport: tool='{name}'")
            tool_result = await mcp_client.call_tool(name, clean_args)

        print(" -> MCP Tool Execution Result Preview:")
        preview = tool_result.strip().splitlines()
        for line in preview[:6]:
            print(f"    {line}")
        if len(preview) > 6:
            print(f"    ... ({len(preview) - 6} more lines)")

        # Robust tool message formatting with role, name, and tool_call_id
        messages.append({
            "role": "tool",
            "name": name,
            "tool_call_id": call_id,
            "content": tool_result,
        })

        tool_trace.append({
            "call_id": call_id,
            "tool": name,
            "arguments": args,
            "result_preview": tool_result[:300],
        })

    print("\n[Step 3: Grounded LLM Response]")
    final_response = ollama.chat(
        model=model_name,
        messages=messages,
    )
    answer = final_response["message"]["content"]
    print(answer)

    return {
        "question": question,
        "tool_trace": tool_trace,
        "final_answer": answer,
    }


def verify_negative_response(response_text: str) -> bool:
    """
    Programmatic assertion: verifies that the assistant correctly recognized
    that the requested amenity (parking/EV charging) is not available or unmentioned.
    """
    unavailability_markers = [
        "not available",
        "no private parking",
        "no car park",
        "no ev charging",
        "does not have",
        "does not offer",
        "unavailable",
        "not mention",
        "no dedicated",
        "closest option",
    ]
    lower = response_text.lower()
    return any(marker in lower for marker in unavailability_markers)


async def run_pipeline():
    target_model = get_available_model()
    print("=" * 80)
    print(f"PART C: MCP-GROUNDED LLM + RAG PIPELINE (Model: {target_model})")
    print("=" * 80)

    # 5 Questions on diverse café topics
    test_questions = [
        "What are the cafe opening hours on Saturday and Sunday?",
        "Can I bring my dog and sit outside on the terrace?",
        "Which food options are suitable for someone following a fully plant-based diet?",
        "Can I work on my laptop, and where can I find Wi-Fi and power sockets?",
        "Which menu items are suitable for someone avoiding wheat or gluten?",
    ]

    async with MCPClientManager() as mcp_client:
        print(f"Connected to MCP Server: {mcp_client.server_info.name if mcp_client.server_info else 'CafeKnowledgeServer'}")

        for q in test_questions:
            await answer_question_with_rag(q, target_model, mcp_client)

        # Negative test question (unanswerable from café documents)
        print("\n" + "#" * 80)
        print("NEGATIVE TEST: TESTING QUESTION NOT PRESENT IN CAFÉ DOCUMENTS")
        print("#" * 80)
        negative_question = "Do you have dedicated customer parking or electric vehicle (EV) charging stations on site?"
        neg_result = await answer_question_with_rag(negative_question, target_model, mcp_client)

        print("\n" + "=" * 80)
        print("PROGRAMMATIC ASSERTION & SAFEGUARD ASSESSMENT")
        print("=" * 80)

        # Execute programmatic assertion
        is_safe = verify_negative_response(neg_result["final_answer"])
        print(f"Negative Question Answer: \"{neg_result['final_answer']}\"")
        print(f"Programmatic Assertion Result (is_safe): {is_safe}")

        assert is_safe, (
            "CRITICAL: Negative test failed! Assistant fabricated unconfirmed details "
            f"instead of identifying unavailability: {neg_result['final_answer']}"
        )
        print(">> ASSERTION PASSED: Assistant correctly stated that parking/EV charging is unavailable/not offered on site.")


def main():
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
