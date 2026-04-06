"""
Test Client for MAF Multi-Agent System
=======================================

This script sends test queries to the Multi-Agent System and prints
all AG-UI events streamed via SSE in real-time.

Two endpoint modes:
  1. /api/agent/run         - Built-in AG-UI adapter (standard protocol)
  2. /api/agent/run/verbose - Custom verbose SSE (maximum observability)

Usage:
  # First, start the server:
  python main.py

  # Then run this test client:
  python test_client.py
"""

import asyncio
import json
import sys

import httpx
from httpx_sse import aconnect_sse


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "http://localhost:8000"
VERBOSE_ENDPOINT = f"{BASE_URL}/api/agent/run/verbose"

# Test queries that exercise different agent capabilities
TEST_QUERIES = [
    # Math-only query → Chat Agent → Orchestrator → Math Agent
    "What is 25 multiplied by 4, then add 17 to the result?",

    # String-only query → Chat Agent → Orchestrator → String Agent
    "Convert 'hello world' to uppercase and then reverse it.",

    # Mixed query → Chat Agent → Orchestrator → both Math & String Agents
    "Add 10 and 20, then convert the result to uppercase text.",
]


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 72)
    print(f"  {text}")
    print("=" * 72)


def print_event(event_type: str, data: dict) -> None:
    """Pretty-print an AG-UI event."""
    # Color codes for different event types
    colors = {
        "RUN_STARTED": "\033[92m",      # Green
        "RUN_FINISHED": "\033[92m",     # Green
        "RUN_ERROR": "\033[91m",        # Red
        "STEP_STARTED": "\033[94m",     # Blue
        "STEP_FINISHED": "\033[94m",    # Blue
        "TEXT_MESSAGE_START": "\033[93m",    # Yellow
        "TEXT_MESSAGE_CONTENT": "\033[97m",  # White
        "TEXT_MESSAGE_END": "\033[93m",      # Yellow
        "TOOL_CALL_START": "\033[95m",  # Magenta
        "TOOL_CALL_ARGS": "\033[95m",   # Magenta
        "TOOL_CALL_END": "\033[95m",    # Magenta
        "TOOL_CALL_RESULT": "\033[96m", # Cyan
        "CUSTOM": "\033[33m",           # Orange
    }
    reset = "\033[0m"
    color = colors.get(event_type, "\033[97m")

    print(f"  {color}[{event_type}]{reset}", end="")

    # Print relevant fields based on event type
    if event_type == "TEXT_MESSAGE_CONTENT":
        delta = data.get("delta", "")
        print(f" {delta}", end="")
    elif event_type == "TOOL_CALL_START":
        print(f" tool={data.get('tool_call_name', '?')}")
    elif event_type == "TOOL_CALL_ARGS":
        print(f" args={data.get('delta', '{}')}")
    elif event_type == "TOOL_CALL_RESULT":
        content = data.get("content", "")
        # Truncate long results for display
        if len(content) > 200:
            content = content[:200] + "..."
        print(f" result={content}")
    elif event_type == "STEP_STARTED":
        print(f" → {data.get('step_name', '?')}")
    elif event_type == "STEP_FINISHED":
        print(f" ✓ {data.get('step_name', '?')}")
    elif event_type == "CUSTOM":
        print(f" name={data.get('name', '?')}: {json.dumps(data.get('value', {}), indent=2)}")
    elif event_type in ("RUN_STARTED", "RUN_FINISHED"):
        print(f" run_id={data.get('run_id', '?')[:8]}...")
    elif event_type == "RUN_ERROR":
        print(f" ❌ {data.get('message', 'Unknown error')}")
    else:
        print()


async def test_verbose_endpoint(query: str) -> None:
    """Test the verbose SSE endpoint with a query.

    Sends a POST request and streams all AG-UI events in real-time.
    """
    print_header(f"QUERY: {query}")
    print(f"  → Endpoint: POST {VERBOSE_ENDPOINT}")
    print(f"  → Streaming AG-UI events...\n")

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            async with aconnect_sse(
                client,
                "POST",
                VERBOSE_ENDPOINT,
                json={"query": query},
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    if sse.data:
                        try:
                            data = json.loads(sse.data)
                            event_type = data.get("type", "UNKNOWN")
                            print_event(event_type, data)
                        except json.JSONDecodeError:
                            print(f"  [RAW] {sse.data}")

        except httpx.ConnectError:
            print("\n  ❌ Could not connect to the server!")
            print("  → Make sure the server is running: python main.py")
            sys.exit(1)
        except Exception as e:
            print(f"\n  ❌ Error: {e}")

    print("\n  ── Stream ended ──")


async def test_health() -> None:
    """Check the server health endpoint."""
    print_header("HEALTH CHECK")
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{BASE_URL}/health")
            health = resp.json()
            print(f"  Status:     {health.get('status', '?')}")
            print(f"  Framework:  {health.get('framework', '?')}")
            print(f"  Agents:     {', '.join(health.get('agents', []))}")
            print(f"  Protocols:  {', '.join(health.get('protocols', []))}")
            print(f"  Model:      {health.get('model', '?')}")
        except httpx.ConnectError:
            print("  ❌ Server not reachable at", BASE_URL)
            print("  → Start the server first: python main.py")
            sys.exit(1)


async def main():
    """Run all test queries against the multi-agent system."""
    print("\n" + "━" * 72)
    print("  MAF Multi-Agent System — Test Client")
    print("  Protocols: MCP + A2A + AG-UI")
    print("━" * 72)

    # 1. Health check
    await test_health()

    # 2. Run test queries (use first query by default, or all with --all flag)
    queries = TEST_QUERIES if "--all" in sys.argv else TEST_QUERIES[:1]

    for query in queries:
        await test_verbose_endpoint(query)

    print_header("ALL TESTS COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
