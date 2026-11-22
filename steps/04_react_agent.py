"""Part D (Steps 16-20): Iterative ReAct Agent Demonstration and Challenge Request."""

import json
import sys
from wandering_kettle_assistant.llm_client import get_available_model
from wandering_kettle_assistant.react_agent import ReActAgent


def main():
    target_model = get_available_model()
    print("=" * 80)
    print("PART D: ITERATIVE REACT AGENT EXECUTION")
    print(f"LLM Engine: {target_model}")
    print("=" * 80)

    agent = ReActAgent(model_name=target_model, max_steps=6)

    # -----------------------------------------------------------------------
    # Case 1: The Multi-Step Challenge Request
    # -----------------------------------------------------------------------
    challenge_request = (
        "I’m planning to visit the café with my cat. Can I sit outside, "
        "what food options are available, and are there any allergy considerations I should know about?"
    )

    print("\n" + "#" * 80)
    print("EXPERIMENT 1: CHALLENGE REQUEST (MULTI-STEP ITERATIVE REACT LOOP)")
    print(f"Prompt: {challenge_request}")
    print("#" * 80)

    result_1 = agent.run(challenge_request)

    print("\n" + "=" * 80)
    print("CHALLENGE REQUEST FINAL SYNTHESIZED ANSWER:")
    print("=" * 80)
    print(result_1["final_answer"])
    print(f"\nTotal Iterations Taken: {result_1['iterations']}")

    # -----------------------------------------------------------------------
    # Case 2: Simple Query (One-Shot / Rapid Stop)
    # -----------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("EXPERIMENT 2: FOCUSED QUERY (RAPID STOP CASE)")
    print("#" * 80)
    simple_request = "What are the café opening hours on Sunday?"
    result_2 = agent.run(simple_request)

    print("\n" + "=" * 80)
    print("FOCUSED QUERY FINAL ANSWER:")
    print("=" * 80)
    print(result_2["final_answer"])
    print(f"\nTotal Iterations Taken: {result_2['iterations']}")

    # -----------------------------------------------------------------------
    # Technical Reflection & Comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TECHNICAL EXPLANATION & AGENT LIMITATION ANALYSIS")
    print("=" * 80)
    print("""
1. One-Shot Tool Call vs. Iterative ReAct Loop:
   - One-shot Tool Call: The LLM makes a single speculative tool decision based on the initial prompt,
     receives the observation, and immediately writes the final answer. If the query requires multiple
     complementary domains (e.g. pet policy + food options + allergen restrictions), a one-shot call
     fails to retrieve all facets or hallucinated missing aspects.
   - Iterative ReAct Loop: The agent operates in an explicit loop (Thought -> Action -> Action Input -> Observation).
     At each step, it inspects the newly retrieved observation, reasons about what is still missing,
     and decides whether another tool invocation is required or if sufficient evidence has been gathered.

2. Loop Components:
   - Reasoning/Decision: LLM generates 'Thought' explaining the missing information and chooses 'Action'.
   - Action: Application-level validation and execution of the selected tool ('Action Input').
   - Observation: Structured response and citation evidence fed back into conversation history.

3. Agent Limitations & Mitigation:
   - Context Growth: As multiple tool calls occur, the observation payloads accumulate in the prompt context.
     * Mitigation: Summarize or extract key sentences before appending observations, or enforce strict top_k limits.
   - Repeated Calls / Loops: Small LLMs can get stuck issuing slight variations of the same query.
     * Mitigation: Deduplication safeguards that track previous (tool, arguments) pairs and abort duplicates.
""")


if __name__ == "__main__":
    main()
