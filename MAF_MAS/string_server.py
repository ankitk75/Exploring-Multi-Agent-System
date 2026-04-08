# =============================================================================
# MCP String Tool Server
# =============================================================================
# Exposes string manipulation tools (to_uppercase, to_lowercase, reverse)
# via the Model Context Protocol.
# Launched as a subprocess by the main agent via MCPStdioTool.
# =============================================================================

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("StringToolServer")


# ── Tool Definitions ─────────────────────────────────────────────────────────


@mcp.tool()
def to_uppercase(text: str) -> str:
    """Convert the input string from lowercase (or mixed case) to fully uppercase."""
    return text.upper()


@mcp.tool()
def to_lowercase(text: str) -> str:
    """Convert the input string from uppercase (or mixed case) to fully lowercase."""
    return text.lower()


@mcp.tool()
def reverse(text: str) -> str:
    """Reverse the input string and return the result."""
    return text[::-1]


# ── Server Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
