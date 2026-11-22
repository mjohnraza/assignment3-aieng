"""Root FastMCP server entry point for Wandering Kettle Café."""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from wandering_kettle_assistant.server import mcp, main

if __name__ == "__main__":
    main()
