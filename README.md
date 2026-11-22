# Wandering Kettle Knowledge Assistant
### MCP • RAG • ReAct Agentic Pipeline

A production-grade AI engineering system featuring semantic vector retrieval with ChromaDB, Model Context Protocol (MCP) server & client integration over Stdio transport, LLM-grounded tool use, and an iterative ReAct reasoning-and-acting agent loop.

---

## 🏛️ End-to-End Architecture

```
User Question
    │
    ▼
Ollama LLM (Model: qwen2.5 / llama3.2)
    │  [Generates Thought & Tool Selection]
    ▼
Python Application Layer
    │  [Validates Arguments against Tool Schema: search, lookup_cafe_policy, search_menu]
    ▼
MCP Client (ClientSession.call_tool() over Stdio transport)
    │  [Standardized JSON-RPC 2.0 Request Frame]
    ▼
MCP Server (CafeKnowledgeServer / FastMCP)
    │  [Dispatches to ChromaDB Vector Store]
    ▼
ChromaDB Persistent Store (Collection: cafe_knowledge_overlapping)
    │  [Cosine similarity, character offset metadata, citation tracking]
    ▼
MCP Tool Observation (with tool_name and tool_call_id)
    │  [Appended back to conversational context]
    ▼
Next Reasoning Turn or Grounded Final Answer
```

---

## 📂 Project Structure

```
wandering-kettle-assistant/
├── chroma_db/                          # Persistent ChromaDB vector database
├── data/
│   └── documents/                      # Ground-truth café text documents
│       ├── allergens.txt
│       ├── faq.txt
│       ├── hours_access.txt
│       ├── menu_drinks.txt
│       ├── menu_food.txt
│       ├── staff_handbook.txt
│       └── story.txt
├── server.py                           # FastMCP / MCPServer root entry point
├── steps/
│   ├── 01_chromadb.py                  # Part A: Baseline line-based indexing
│   ├── 02_raw_jsonrpc.py               # Part B: Live raw JSON-RPC 2.0 client-server exchange
│   ├── 02_mcp_server.py                # Part B: MCP server runner script
│   ├── 02_mcp_client.py                # Part B: MCP client SDK demonstration (tools/list & tools/call)
│   ├── 03_llm_rag.py                   # Part C: LLM + RAG via MCP ClientSession & negative assertion
│   └── 04_react_agent.py               # Part D: Full Iterative ReAct loop via MCP Client
├── src/
│   └── wandering_kettle_assistant/
│       ├── __init__.py
│       ├── citations.py                # Citation tracking, metadata & formatters
│       ├── ingest.py                   # Improved 300-char overlapping chunk ingestion
│       ├── llm_client.py               # Ollama client and ReAct parsing helpers
│       ├── mcp_client.py               # MCPClientManager (ClientSession over Stdio)
│       ├── react_agent.py              # Native ReAct agent with MCP execution & safeguards
│       ├── retrieval.py                # ChromaDB vector retrieval & policy search
│       ├── server.py                   # FastMCP / MCPServer tool declarations
│       └── validation.py               # Schema validation for search, policy, and menu args
├── tests/
│   ├── test_react_agent.py             # Unit tests for ReAct multi-step, safeguards, & MCP
│   ├── test_retrieval.py               # Unit tests for retrieval & citations
│   └── test_validation.py              # Unit tests for schema & boundary validation
├── experiments/
│   └── compare_retrieval.py            # Quantitative benchmark: Baseline vs Overlapping
├── evidence/
│   └── part_a_retrieval_comparison.json # Benchmark metrics (P@K, R@K, MRR) & manual assessments
├── pyproject.toml
└── uv.lock
```

---

## 🚀 Execution Guide

Run all steps using `uv run python <path>`:

### Part A: RAG Pipeline & Overlapping Chunking
1. **Run Improved Overlapping Ingestion:**
   ```bash
   uv run python src/wandering_kettle_assistant/ingest.py
   ```
2. **Run Baseline Ingestion (for comparison):**
   ```bash
   uv run python steps/01_chromadb.py
   ```
3. **Execute Comparative Evaluation Benchmark:**
   ```bash
   uv run python experiments/compare_retrieval.py
   ```
   *Evaluates both Source-Level and Passage-Level Precision@K, Recall@K, and Mean Reciprocal Rank (MRR), with detailed manual assessments in `evidence/part_a_retrieval_comparison.json`.*

#### ⚖️ Retrieval Evaluation Trade-Off Analysis
| Metric | Baseline (Line-based) | Overlapping (300 char / 50 overlap) | Engineering Rationale |
| :--- | :---: | :---: | :--- |
| **Source Recall@3** | 0.917 | **1.000** | Perfect source coverage: all relevant policy documents captured in top-3. |
| **Source Precision@3** | **0.945** | 0.778 | Baseline matches isolated lines; overlapping windows include contiguous sentences. |
| **Passage Recall@3** | 0.750 | **1.000** | Overlapping chunks guarantee all necessary information sources have answering passages. |
| **Passage Precision@3**| 0.389 | **0.611** | Overlapping chunks provide higher factual answer density (+57% improvement). |
| **Mean Reciprocal Rank** | 0.667 | **1.000** | First relevant answer rank is preserved at rank 1 across all queries. |

> **Key Trade-off:** Overlapping chunking trades off raw source precision for complete contextual recall and contiguous semantic paragraphs. In safety-critical café RAG (e.g. allergen warnings and pet restrictions), preventing split-sentence truncation is prioritized over raw line precision. Evidence is serialized in `evidence/part_a_retrieval_comparison.json`.

### Part B: MCP Fundamentals
1. **Live Raw JSON-RPC 2.0 Protocol Exchange (Steps 6 & 7):**
   ```bash
   uv run python steps/02_raw_jsonrpc.py
   ```
   *Spawns `server.py` as a subprocess and performs an actual raw line-delimited JSON-RPC 2.0 handshake (`initialize`), notification (`notifications/initialized`), tool discovery (`tools/list`), and execution (`tools/call`).*
2. **MCP SDK Client & Server Exchange (Steps 8–11):**
   ```bash
   uv run python steps/02_mcp_client.py
   ```
   *Connects via MCP `ClientSession` over Stdio transport, discovering and invoking `search`, `lookup_cafe_policy`, and `search_menu`.*

### Part C: LLM + RAG Integration (Steps 12–15)
```bash
uv run python steps/03_llm_rag.py
```
- Implements the complete sequence: `User question → LLM tool decision → MCP Client invocation (ClientSession.call_tool()) → retrieved context → LLM response`.
- Evaluates 5 café questions across diverse topics.
- **Dual-Aspect Programmatic Negative Test Assertion:** Independently verifies that the assistant accurately states both that on-site parking is unavailable (or refers to the coin lot) AND that EV charging is unavailable/unmentioned, guarding against partial hallucinations.
- **MCP Error Flag Inspection:** Inspects `result.is_error` / `result.isError` directly on MCP responses.

### Part D: Iterative ReAct Agent (Steps 16–20)
```bash
uv run python steps/04_react_agent.py
```
- Executes the multi-step challenge request:
  > *"I’m planning to visit the café with my cat. Can I sit outside, what food options are available, and are there any allergy considerations I should know about?"*
- Runs an explicit `while steps < MAX_STEPS:` loop in Python with tool calls executed strictly through the MCP client (`ClientSession.call_tool()`).
- Maintains robust conversation history with `role: "tool"`, `name`, and `tool_call_id`.
- **Duplicate-Call Safeguard:** Tracks executed `(tool, argument)` fingerprints across iterations and skips redundant identical calls.
- **Defensive Schema Validation:** Robustly handles malformed non-dictionary inputs (`None`, `list`, primitives) across all validation functions.
- **End-to-End Error Propagation:** Raises `RetrievalError` on vector search/ChromaDB exceptions, captured by FastMCP as `isError=True` and flagged in `ToolObservation.is_error`.
- Validates arguments against schema prior to dispatch and propagates MCP error flags back to conversational history.
- Emits citations and explicitly distinguishes confirmed facts (e.g. dogs permitted on outdoor terrace, assistance animals permitted everywhere, vegan & gluten-free food options) from unconfirmed details (e.g. cats specifically are unmentioned in policy).

### Automated Testing
Run the complete test suite (23 passed):
```bash
uv run pytest
```
```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\exam.DESKTOP-LNCC6CA\Desktop\AA3\wandering-kettle-assistant
configfile: pyproject.toml
plugins: anyio-4.16.0, asyncio-1.5.0
collected 23 items

tests\test_react_agent.py .........                                      [ 39%]
tests\test_retrieval.py ....                                             [ 56%]
tests\test_validation.py ..........                                      [100%]

============================= 23 passed in 11.26s =============================
```
