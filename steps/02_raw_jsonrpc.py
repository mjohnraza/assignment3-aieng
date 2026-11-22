"""Part B (Steps 6 & 7): Actual Raw JSON-RPC 2.0 Client-Server Exchange Demonstration.

Implements an actual raw JSON-RPC 2.0 client communicating over stdin/stdout pipes
with the FastMCP server subprocess without using the high-level MCP SDK client.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from wandering_kettle_assistant.raw_jsonrpc import (
    RobustRawJSONRPCClient,
    JSONRPCError,
    JSONRPCParseError,
    JSONRPCVersionError,
    JSONRPCIdMismatchError,
    JSONRPCResponseError,
    ServerProcessTerminatedError,
    JSONRPCTimeoutError,
    MalformedToolResultError,
    MCPToolExecutionError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = PROJECT_ROOT / "server.py"


def run_raw_jsonrpc_exchange():
    print("=" * 80, flush=True)
    print("PART B: LIVE RAW JSON-RPC 2.0 PROTOCOL EXCHANGE (DEFENSIVE CLIENT)", flush=True)
    print(f"Target Server Subprocess: {SERVER_SCRIPT}", flush=True)
    print("=" * 80, flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-u", str(SERVER_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=0,
        env=env,
    )

    client = RobustRawJSONRPCClient()

    def send_raw_request(req_payload: dict, timeout: float = 10.0) -> dict:
        req_id = req_payload.get("id")
        wire_str = json.dumps(req_payload) + "\n"
        print(f"\n[CLIENT -> SERVER (RAW WIRE JSON)]:\n{wire_str.strip()}", flush=True)
        proc.stdin.write(wire_str)
        proc.stdin.flush()

        response = client.read_next_jsonrpc_frame(proc, expected_id=req_id, timeout=timeout)
        print(f"[SERVER -> CLIENT (VALIDATED JSON-RPC 2.0 FRAME)]:\n{json.dumps(response)}", flush=True)
        return response

    try:
        # -------------------------------------------------------------------
        # 1. MCP Initialization Handshake (initialize)
        # -------------------------------------------------------------------
        print("\n--- 1. Raw MCP Handshake: initialize ---", flush=True)
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {},
                },
                "clientInfo": {
                    "name": "RawJSONRPCClient",
                    "version": "1.0.0",
                },
            },
        }
        init_response = send_raw_request(init_request)
        print("\nParsed Initialization Response:", flush=True)
        print(json.dumps(init_response, indent=2), flush=True)

        # -------------------------------------------------------------------
        # 2. Initialized Notification (notifications/initialized)
        # -------------------------------------------------------------------
        print("\n--- 2. Client Notification: notifications/initialized ---", flush=True)
        notif = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        notif_str = json.dumps(notif) + "\n"
        print(f"[CLIENT -> SERVER (RAW WIRE NOTIFICATION)]:\n{notif_str.strip()}", flush=True)
        proc.stdin.write(notif_str)
        proc.stdin.flush()

        # -------------------------------------------------------------------
        # 3. Tool Discovery: tools/list
        # -------------------------------------------------------------------
        print("\n--- 3. Tool Discovery: tools/list ---", flush=True)
        tools_list_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
        tools_list_response = send_raw_request(tools_list_request)
        discovered_tools = tools_list_response.get("result", {}).get("tools", [])
        print(f"\nDiscovered {len(discovered_tools)} tools via raw tools/list:", flush=True)
        for t in discovered_tools:
            print(f"  - {t['name']}: {t.get('description', '').splitlines()[0]}", flush=True)

        # -------------------------------------------------------------------
        # 4. Tool Invocation: tools/call
        # -------------------------------------------------------------------
        print("\n--- 4. Tool Invocation: tools/call ---", flush=True)
        call_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "lookup_cafe_policy",
                "arguments": {
                    "topic": "pet_policy",
                },
            },
        }
        call_response = send_raw_request(call_request)
        print("\nParsed Tool Call Response:", flush=True)
        print(json.dumps(call_response, indent=2), flush=True)

        extracted_text = client.extract_and_validate_tool_result(call_response)
        print(f"\nExtracted and Validated Tool Content:\n{extracted_text}", flush=True)

    finally:
        RobustRawJSONRPCClient.close_reader(proc)
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            pass

    print("\n" + "=" * 80, flush=True)
    print("RAW JSON-RPC 2.0 PROTOCOL STUDY & ARCHITECTURAL SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print("""
1. Initialization Exchange ('initialize'):
   Negotiates protocol version and exchanges capabilities between client and server
   before any commands or tool invocations are transmitted.

2. Initialized Notification ('notifications/initialized'):
   Informs the server that the client has received the initialization response
   and is ready to start regular operation.

3. Tool Discovery ('tools/list'):
   Standardized endpoint where the server declares available tools and parameter JSON schemas.
   The client passes these schemas to the LLM for tool selection.

4. Tool Invocation ('tools/call'):
   Standardized JSON-RPC request containing the tool name and validated arguments.
   The server executes the retrieval and returns structured content with an isError boolean.

5. MCP vs. Underlying Functionality:
   MCP standardizes transport and schema discovery across diverse systems.
   It does not perform document retrieval itself; the server implementation delegates
   the actual retrieval to ChromaDB / vector embeddings.
""", flush=True)


if __name__ == "__main__":
    run_raw_jsonrpc_exchange()
