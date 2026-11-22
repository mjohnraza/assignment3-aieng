"""Part D (Steps 16-20): Iterative ReAct Agent Execution via MCP Client.

Demonstrates:
1. Multi-step ReAct reasoning loop with tool calls routed through MCP ClientSession.call_tool().
2. The Challenge Request: Cat visit, outdoor terrace seating, food options, and allergy notes.
3. Rapid stopping on single-domain query.
4. Robust tool-response history tracking with explicit tool_call_id and name.
5. In-depth technical reflection and limitation analysis.
"""

import asyncio
import json
import sys
from wandering_kettle_assistant.llm_client import get_available_model
from wandering_kettle_assistant.mcp_client import MCPClientManager
from wandering_kettle_assistant.react_agent import ReActAgent


async def run_react_experiments():
    target_model = get_available_model()
    print("=" * 80)
    print("PART D: ITERATIVE REACT AGENT EXECUTION VIA MCP CLIENT")
    print(f"LLM Engine: {target_model}")
    print("Transport: Model Context Protocol (MCP) Stdio ClientSession")
    print("=" * 80)

    async with MCPClientManager() as mcp_client:
        agent = ReActAgent(model_name=target_model, max_steps=6, mcp_client=mcp_client)

        # -------------------------------------------------------------------
        # Experiment 1: The Multi-Step Challenge Request
        # -------------------------------------------------------------------
        challenge_request = (
            "I’m planning to visit the café with my cat. Can I sit outside, "
            "what food options are available, and are there any allergy considerations I should know about?"
        )

        print("\n" + "#" * 80)
        print("EXPERIMENT 1: CHALLENGE REQUEST (MULTI-STEP ITERATIVE REACT LOOP)")
        print(f"Prompt: {challenge_request}")
        print("#" * 80)

        result_1 = await agent.arun(challenge_request, mcp_client=mcp_client)

        print("\n" + "=" * 80)
        print("CHALLENGE REQUEST FINAL SYNTHESIZED ANSWER:")
        print("=" * 80)
        print(result_1["final_answer"])
        print(f"\nTotal Iterations Taken: {result_1['iterations']}")
        print(f"Status: {result_1['status']}")

        # -------------------------------------------------------------------
        # Experiment 2: Simple Query (One-Shot / Rapid Stop Case)
        # -------------------------------------------------------------------
        print("\n" + "#" * 80)
        print("EXPERIMENT 2: FOCUSED QUERY (RAPID STOP CASE)")
        print("#" * 80)
        simple_request = "What are the café opening hours on Sunday?"
        result_2 = await agent.arun(simple_request, mcp_client=mcp_client)

        print("\n" + "=" * 80)
        print("FOCUSED QUERY FINAL ANSWER:")
        print("=" * 80)
        print(result_2["final_answer"])
        print(f"\nTotal Iterations Taken: {result_2['iterations']}")
        print(f"Status: {result_2['status']}")

    # -----------------------------------------------------------------------
    # Technical Reflection & Comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TECHNICAL EXPLANATION & ARCHITECTURAL SUMMARY")
    print("=" * 80)
    print("""
1. One-Shot Tool Call vs. Iterative ReAct Loop:
   - One-Shot Tool Call: The LLM makes a single speculative tool invocation decision,
     receives the observation, and immediately writes the final answer. If the query requires multiple
     complementary domains (e.g. pet policy + food options + allergen restrictions), a one-shot call
     fails to retrieve all facets or hallucinates missing aspects.
   - Iterative ReAct Loop: The agent operates in an explicit loop (Thought -> Action -> Action Input -> Observation).
     At each step, it inspects the newly retrieved observation, reasons about what is still missing,
     and decides whether another tool invocation is required or if sufficient evidence has been gathered.

2. Loop Components:
   - Reasoning/Decision: LLM generates 'Thought' explaining the missing information and chooses 'Action'.
   - Action: Application-level validation against the exact tool argument schema before sending.
   - MCP Tool Invocation: The tool is dispatched across stdio transport using MCP ClientSession.call_tool().
   - Observation: Structured response and citation evidence fed back into conversation history with
     corresponding tool_call_id and tool name.

3. Agent Limitations & Mitigation:
   - Context Growth: As multiple tool calls occur, the observation payloads accumulate in the prompt context.
     * Mitigation: Summarize or extract key sentences before appending observations, or enforce strict top_k limits.
   - Repeated Calls / Loops: Small LLMs can get stuck issuing slight variations of the same query.
     * Mitigation: Deduplication safeguards that track previous (tool, arguments) pairs and abort duplicates.
""")


def main():
    asyncio.run(run_react_experiments())


if __name__ == "__main__":
    main()
