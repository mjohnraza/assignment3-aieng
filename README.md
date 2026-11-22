# Wandering Kettle Knowledge Assistant
### MCP • RAG • ReAct Agentic Pipeline

A production-grade AI engineering assignment implementation featuring semantic retrieval with ChromaDB, Model Context Protocol (MCP) tool exposure over Stdio transport, LLM-grounded tool use, and an iterative ReAct reasoning-and-acting loop.

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
│   ├── 01_chromadb.py                  # Part A: ChromaDB indexing & overlapping chunks
│   ├── 02_raw_jsonrpc.py               # Part B: Raw JSON-RPC 2.0 protocol exchange demonstration
│   ├── 02_mcp_server.py                # Part B: MCP server runner
│   ├── 02_mcp_client.py                # Part B: MCP client (tools/list & tools/call via Stdio)
│   ├── 03_llm_rag.py                   # Part C: LLM + RAG single-turn tool calling & safeguards
│   └── 04_react_agent.py               # Part D: Full Iterative ReAct loop & challenge query
├── src/
│   └── wandering_kettle_assistant/
│       ├── __init__.py
│       ├── citations.py                # Citation tracking, metadata & formatters
│       ├── ingest.py                   # Document chunking with character offsets
│       ├── llm_client.py               # Ollama client and ReAct parsing helpers
│       ├── react_agent.py              # Native while-loop ReAct agent implementation
│       ├── retrieval.py                # ChromaDB vector retrieval & policy search
│       ├── server.py                   # FastMCP / MCPServer tool declarations
│       └── validation.py               # Pydantic/dataclass argument validation schemas
├── tests/
│   ├── test_react_agent.py             # Unit tests for ReAct parsing & agent
│   ├── test_retrieval.py               # Unit tests for retrieval & citations
│   └── test_validation.py              # Unit tests for argument & boundary validation
├── pyproject.toml
└── uv.lock
```

---

## 🚀 Execution Guide

Run all steps using `uv run python <path>`:

### Part A: Baseline & Overlapping Chunk Retrieval
```bash
uv run python steps/01_chromadb.py
uv run python experiments/compare_retrieval.py
```

### Part B: Expose Retrieval Through MCP
1. **Raw JSON-RPC 2.0 Demonstration (Steps 6 & 7):**
   ```bash
   uv run python steps/02_raw_jsonrpc.py
   ```
2. **MCP SDK Client & Server Exchange (Steps 8–11):**
   ```bash
   uv run python steps/02_mcp_client.py
   ```
   *Spawns `server.py` over Stdio transport, runs `session.initialize()`, invokes `tools/list`, and calls `search`, `lookup_cafe_policy`, and `search_menu`.*

### Part C: Connect the LLM and RAG (Steps 12–15)
```bash
uv run python steps/03_llm_rag.py
```
*Evaluates the LLM tool-calling flow on 5 café questions plus a negative unanswerable test (EV charging / parking) verifying grounded answers and safeguards.*

### Part D: Iterative ReAct Agent (Steps 16–20)
```bash
uv run python steps/04_react_agent.py
```
*Executes the multi-step challenge request:*
> *"I’m planning to visit the café with my cat. Can I sit outside, what food options are available, and are there any allergy considerations I should know about?"*
*Demonstrates multi-turn tool invocation, validation, citation tracking, and distinction between confirmed information and unconfirmed details (e.g. dogs explicitly permitted on terrace vs cats not listed).*

### Run Automated Tests
```bash
uv run pytest
```
