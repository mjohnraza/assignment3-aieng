"""FastMCP / MCPServer exposing Wandering Kettle Café Knowledge Base."""

import sys

try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

from wandering_kettle_assistant.retrieval import (
    search_knowledge_base,
    lookup_policy,
    search_menu as get_menu_info,
)

mcp = FastMCP("CafeKnowledgeServer")


@mcp.tool()
def search(query: str, top_k: int = 5) -> str:
    """Search the Wandering Kettle café knowledge base.

    Use this tool to find information about:
    - Menu items, seasonal drinks, pastry options, and pricing
    - Allergen warnings, dairy alternatives, and dietary accommodations
    - Brewing guides, roast profiles, espresso extraction, and coffee origins
    - Café hours, seating policies, Wi-Fi access, and location details

    Args:
        query: Search query covering café topics, drinks, ingredients, or policies.
        top_k: Maximum number of relevant document passages to retrieve.

    Returns:
        Formatted string containing retrieved passages and source metadata.
    """
    return search_knowledge_base(query=query, top_k=top_k)


@mcp.tool()
def lookup_cafe_policy(topic: str) -> str:
    """Lookup official café policies including pet rules, outdoor seating, Wi-Fi, and hours.

    Args:
        topic: Specific policy area such as 'pet_policy', 'outdoor_seating', 'wifi', or 'hours'.

    Returns:
        Documented café policy guidelines and constraints.
    """
    return lookup_policy(topic=topic)


@mcp.tool()
def search_menu(category: str = "food", include_allergens: bool = True) -> str:
    """Search café food and drink menu items with detailed allergen notes.

    Args:
        category: Menu category such as 'food', 'drinks', 'pastry', or 'all'.
        include_allergens: Whether to include cross-contamination and allergen warnings.

    Returns:
        Available items matching the category and allergen considerations.
    """
    return get_menu_info(category=category, include_allergens=include_allergens)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
