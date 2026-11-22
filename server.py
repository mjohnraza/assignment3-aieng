"""Root FastMCP server entry point for Wandering Kettle Café."""

import os
if not os.environ.get("CI") and "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

from wandering_kettle_assistant.server import mcp, main

if __name__ == "__main__":
    main()
