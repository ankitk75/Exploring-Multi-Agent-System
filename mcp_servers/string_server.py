"""
MCP String Tool Server
======================
Microsoft Agent Framework (MAF) - MCP Tool Layer

This MCP server exposes string manipulation tools via the Model Context
Protocol using stdio transport. The String Agent in our multi-agent system
consumes these tools purely through the MCP protocol.

Tools provided:
  - convert_to_uppercase(text): Converts text to uppercase
  - convert_to_lowercase(text): Converts text to lowercase
  - reverse_string(text): Reverses the input string

Usage:
  This server is launched as a subprocess by the main application using
  StdioServerParameters. It communicates via stdin/stdout following the
  MCP JSON-RPC protocol.
"""

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Initialize the MCP server instance with a descriptive name.
# ---------------------------------------------------------------------------
mcp = FastMCP("StringToolServer")


@mcp.tool()
def convert_to_uppercase(text: str) -> str:
    """Convert the given text to uppercase.

    Args:
        text: The input string to convert.

    Returns:
        The input string converted to all uppercase characters.
    """
    result = text.upper()
    print(f"[MCP StringServer] convert_to_uppercase('{text}') = '{result}'")
    return result


@mcp.tool()
def convert_to_lowercase(text: str) -> str:
    """Convert the given text to lowercase.

    Args:
        text: The input string to convert.

    Returns:
        The input string converted to all lowercase characters.
    """
    result = text.lower()
    print(f"[MCP StringServer] convert_to_lowercase('{text}') = '{result}'")
    return result


@mcp.tool()
def reverse_string(text: str) -> str:
    """Reverse the given string.

    Args:
        text: The input string to reverse.

    Returns:
        The input string with characters in reverse order.
    """
    result = text[::-1]
    print(f"[MCP StringServer] reverse_string('{text}') = '{result}'")
    return result


# ---------------------------------------------------------------------------
# Entry point: Run the MCP server using stdio transport.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
