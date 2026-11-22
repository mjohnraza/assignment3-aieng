"""Part B (Steps 6 & 7): Study and demonstrate the raw JSON-RPC 2.0 MCP protocol."""

import json

print("=" * 80)
print("PART B: RAW JSON-RPC 2.0 PROTOCOL STUDY & DEMONSTRATION")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. MCP Initialization Exchange (Handshake)
# ---------------------------------------------------------------------------
print("\n--- 1. MCP Initialization Request (Client -> Server) ---")
init_request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "roots": {"listChanged": True},
            "sampling": {}
        },
        "clientInfo": {
            "name": "WanderingKettleClient",
            "version": "1.0.0"
        }
    }
}
print(json.dumps(init_request, indent=2))

print("\n--- 1. MCP Initialization Response (Server -> Client) ---")
init_response = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {},
            "prompts": {}
        },
        "serverInfo": {
            "name": "CafeKnowledgeServer",
            "version": "0.1.0"
        }
    }
}
print(json.dumps(init_response, indent=2))

# ---------------------------------------------------------------------------
# 2. Tool Discovery: tools/list
# ---------------------------------------------------------------------------
print("\n--- 2. Tool Discovery Request: tools/list (Client -> Server) ---")
tools_list_request = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {}
}
print(json.dumps(tools_list_request, indent=2))

print("\n--- 2. Tool Discovery Response: tools/list (Server -> Client) ---")
tools_list_response = {
    "jsonrpc": "2.0",
    "id": 2,
    "result": {
        "tools": [
            {
                "name": "search",
                "description": "Search the Wandering Kettle café knowledge base.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "lookup_cafe_policy",
                "description": "Lookup official café policies including pet rules, outdoor seating, Wi-Fi, and hours.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Policy area"}
                    },
                    "required": ["topic"]
                }
            }
        ]
    }
}
print(json.dumps(tools_list_response, indent=2))

# ---------------------------------------------------------------------------
# 3. Tool Invocation: tools/call
# ---------------------------------------------------------------------------
print("\n--- 3. Tool Invocation Request: tools/call (Client -> Server) ---")
# Standardized JSON-RPC 2.0 tool invocation request sent from MCP client to server
json_rpc_request = {
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
        "name": "search",
        "arguments": {
            "query": "matcha latte options",
            "top_k": 2
        }
    }
}
print("=== MCP tools/call Request ===")
print(json.dumps(json_rpc_request, indent=2))

print("\n--- 3. Tool Invocation Response: tools/call (Server -> Client) ---")
# Standardized JSON-RPC 2.0 tool invocation response returned by MCP server
json_rpc_response = {
    "jsonrpc": "2.0",
    "id": 3,
    "result": {
        "content": [
            {
                "type": "text",
                "text": "Matcha Latte: Ceremonial grade green tea with steamed milk."
            },
            {
                "type": "text",
                "text": "Iced Matcha Latte: Served over ice with oat milk."
            }
        ],
        "isError": False
    }
}
print("=== MCP tools/call Response ===")
print(json.dumps(json_rpc_response, indent=2))

print("\n" + "=" * 80)
print("EXPLANATION & ARCHITECTURAL SUMMARY")
print("=" * 80)
print("""
1. Initialization Exchange ('initialize'):
   Negotiates protocol version and exchanges capabilities between client and server
   before any commands or tool invocations are transmitted.

2. Tool Discovery ('tools/list'):
   Standardized endpoint where the server declares which tools are available, along with
   their JSON schema descriptions. The client uses this to expose tools to LLMs.

3. Tool Invocation ('tools/call'):
   Standardized JSON-RPC call where the client supplies the tool name and validated arguments.
   The server dispatches the request to the underlying implementation and returns
   standardized content blocks (text/images) with an isError boolean.

4. MCP vs. Underlying Functionality:
   MCP standardizes transport and schema discovery across diverse systems.
   It does not perform document retrieval itself; the server implementation delegates
   the actual retrieval to ChromaDB / vector embeddings.
""")
