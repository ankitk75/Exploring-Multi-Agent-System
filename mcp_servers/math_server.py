"""
MCP Math Tool Server
====================
Microsoft Agent Framework (MAF) - MCP Tool Layer

This MCP server exposes mathematical tools via the Model Context Protocol
using stdio transport. The Math Agent in our multi-agent system consumes
these tools purely through the MCP protocol.

Tools provided:
  - add(a, b): Returns the sum of two numbers
  - multiply(a, b): Returns the product of two numbers
  - divide(a, b): Returns the quotient (with division-by-zero guard)

Usage:
  This server is launched as a subprocess by the main application using
  StdioServerParameters. It communicates via stdin/stdout following the
  MCP JSON-RPC protocol.
"""

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Initialize the MCP server instance with a descriptive name.
# The name helps identify this server in logs and MCP discovery.
# ---------------------------------------------------------------------------
mcp = FastMCP("MathToolServer")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together and return the sum.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum of a and b.
    """
    result = a + b
    print(f"[MCP MathServer] add({a}, {b}) = {result}")
    return result


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The product of a and b.
    """
    result = a * b
    print(f"[MCP MathServer] multiply({a}, {b}) = {result}")
    return result


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide the first number by the second and return the quotient.

    Includes a division-by-zero guard that returns an error message
    instead of raising an exception.

    Args:
        a: The numerator.
        b: The denominator (must not be zero).

    Returns:
        The quotient of a / b, or an error string if b is zero.
    """
    if b == 0:
        return "Error: Division by zero is not allowed."
    result = a / b
    print(f"[MCP MathServer] divide({a}, {b}) = {result}")
    return result


# ---------------------------------------------------------------------------
# Entry point: Run the MCP server using stdio transport.
# When launched via StdioServerParameters, the parent process communicates
# with this server over stdin/stdout using MCP's JSON-RPC protocol.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
