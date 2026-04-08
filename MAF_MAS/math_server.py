# =============================================================================
# MCP Math Tool Server
# =============================================================================
# Exposes math tools (add, multiply, divide) via the Model Context Protocol.
# Launched as a subprocess by the main agent via MCPStdioTool.
# =============================================================================

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MathToolServer")


# ── Tool Definitions ─────────────────────────────────────────────────────────


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together and return the result."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together and return the result."""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide the first number by the second. Returns an error message if dividing by zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


# ── Server Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
