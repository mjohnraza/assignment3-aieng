"""Part B: MCP Server wrapper script for step execution."""

import sys
from wandering_kettle_assistant.server import mcp

if __name__ == "__main__":
    print("Starting CafeKnowledgeServer over stdio transport...", file=sys.stderr)
    mcp.run()
