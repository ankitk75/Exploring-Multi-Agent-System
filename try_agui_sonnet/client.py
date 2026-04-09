"""
client.py  ·  AG-UI event collector for the MAF Multi-Agent System
==================================================================

Sends a single query to the running server, captures every AG-UI
Server-Sent Event, and saves the full trace to a timestamped JSON file.

Usage
-----
    # Make sure server.py is running first.

    python client.py "math: what is (17 * 4) + 9?"
    python client.py "string: reverse 'Hello World' then uppercase it"
    python client.py "magentic: compute 8 times 9 and reverse the result"

    # Specify a custom output path:
    python client.py "math: 100 / 4" --out my_trace.json

    # Point at a non-default server:
    python client.py "string: lowercase ABC" --url http://my-host:9000/

Output JSON structure
---------------------
{
  "query":        "<original query>",
  "route":        "math | string | magentic",
  "server_url":   "http://...",
  "started_at":   "<ISO-8601>",
  "finished_at":  "<ISO-8601>",
  "duration_s":   1.23,
  "event_count":  42,
  "event_summary": {"RUN_STARTED": 1, "TEXT_MESSAGE_CONTENT": 18, ...},
  "events": [
    {"type": "RUN_STARTED",  "runId": "...", "threadId": "...", ...},
    {"type": "STEP_STARTED", ...},
    {"type": "TEXT_MESSAGE_START", ...},
    {"type": "TEXT_MESSAGE_CONTENT", "delta": "The result", ...},
    ...
    {"type": "RUN_FINISHED", ...}
  ]
}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SERVER = "http://127.0.0.1:8888/"
DEFAULT_TIMEOUT = 180.0          # seconds (Magentic runs can take a while)

# ANSI colour helpers for the live console output
_COLORS: dict[str, str] = {
    "RUN_STARTED":            "\033[92m",   # bright green
    "RUN_FINISHED":           "\033[92m",
    "RUN_ERROR":              "\033[91m",   # bright red
    "STEP_STARTED":           "\033[94m",   # bright blue
    "STEP_FINISHED":          "\033[94m",
    "TEXT_MESSAGE_START":     "\033[96m",   # cyan
    "TEXT_MESSAGE_CONTENT":   "\033[97m",   # white
    "TEXT_MESSAGE_END":       "\033[96m",
    "TEXT_MESSAGE_CHUNK":     "\033[97m",
    "TOOL_CALL_START":        "\033[93m",   # yellow
    "TOOL_CALL_ARGS":         "\033[93m",
    "TOOL_CALL_END":          "\033[93m",
    "TOOL_CALL_RESULT":       "\033[33m",   # orange-ish
    "STATE_SNAPSHOT":         "\033[35m",   # magenta
    "STATE_DELTA":            "\033[35m",
    "MESSAGES_SNAPSHOT":      "\033[35m",
    "magentic_orchestrator":  "\033[95m",   # bright magenta
    "group_chat":             "\033[95m",
    "output":                 "\033[97m",
    "REASONING_START":        "\033[36m",   # teal
    "REASONING_END":          "\033[36m",
    "CUSTOM":                 "\033[37m",   # grey
    "RAW":                    "\033[37m",
}
_RESET = "\033[0m"


def _colour(event_type: str, text: str) -> str:
    c = _COLORS.get(event_type, "\033[37m")
    return f"{c}{text}{_RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# Route detection (mirrors the server logic so we can tag the output file)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_route(query: str) -> str:
    low = query.strip().lower()
    if low.startswith("math:"):
        return "math"
    if low.startswith("string:"):
        return "string"
    if low.startswith("magentic:"):
        return "magentic"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# AG-UI request / SSE collector
# ─────────────────────────────────────────────────────────────────────────────

async def collect_events(
    query: str,
    server_url: str,
    timeout: float,
) -> list[dict]:
    """
    POST a RunAgentInput to `server_url`, consume the SSE stream, parse
    every `data: {...}` line as JSON, and return the list of event dicts.

    The AG-UI RunAgentInput schema:
        thread_id  – stable ID for the conversation thread
        run_id     – unique ID for this single execution
        messages   – list of {role, content} message objects
        state      – optional shared state dict (empty here)
    """
    thread_id = f"thread-{uuid.uuid4().hex[:10]}"
    run_id    = f"run-{uuid.uuid4().hex[:10]}"

    payload = {
        "thread_id": thread_id,
        "run_id":    run_id,
        "messages": [
            {"id": f"msg-{uuid.uuid4().hex[:8]}", "role": "user", "content": query}
        ],
        "state": {},
    }

    events: list[dict] = []
    parse_errors       = 0

    async with httpx.AsyncClient(timeout=timeout) as http:
        async with http.stream("POST", server_url, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(
                    f"Server returned HTTP {resp.status_code}: {body.decode()[:200]}"
                )

            async for raw_line in resp.aiter_lines():
                # SSE lines are either:
                #   data: {...json...}
                #   data: [DONE]
                #   (empty lines between events)
                if not raw_line.startswith("data:"):
                    continue

                data = raw_line[len("data:"):].strip()

                if data in ("[DONE]", ""):
                    continue

                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    parse_errors += 1
                    print(f"  ⚠  JSON parse error ({exc}): {data!r:.100}", file=sys.stderr)
                    continue

                events.append(event)

                # ── Live console output ──────────────────────────────────────
                etype = event.get("type", "?")
                label = f"  [{etype}]"

                # For text content show a snippet of the delta
                if etype in ("TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_CHUNK"):
                    delta = event.get("delta", "")
                    print(_colour(etype, f"{label}  {delta!r:.60}"), flush=True)

                # For tool calls show the tool name / args
                elif etype == "TOOL_CALL_START":
                    tool_name = event.get("toolCallName", event.get("tool_name", "?"))
                    print(_colour(etype, f"{label}  tool={tool_name}"), flush=True)

                elif etype == "TOOL_CALL_ARGS":
                    delta = event.get("delta", "")
                    print(_colour(etype, f"{label}  args={delta!r:.60}"), flush=True)

                elif etype == "TOOL_CALL_RESULT":
                    content = event.get("content", "")
                    print(_colour(etype, f"{label}  result={str(content)!r:.60}"), flush=True)

                # For Magentic orchestrator events show the phase
                elif etype == "magentic_orchestrator":
                    phase = (
                        event.get("data", {})
                             .get("event_type", {})
                             .get("name", "?")
                    )
                    print(_colour(etype, f"{label}  phase={phase}"), flush=True)

                # For group-chat events show which agent got the baton
                elif etype == "group_chat":
                    participant = (
                        event.get("data", {}).get("participant_name", "?")
                    )
                    print(_colour(etype, f"{label}  → {participant}"), flush=True)

                # All other events: just show the type
                else:
                    print(_colour(etype, label), flush=True)

    if parse_errors:
        print(f"\n  ⚠  {parse_errors} line(s) could not be parsed as JSON.", file=sys.stderr)

    return events


# ─────────────────────────────────────────────────────────────────────────────
# Save to JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_to_json(
    events:      list[dict],
    query:       str,
    server_url:  str,
    started_at:  str,
    finished_at: str,
    duration_s:  float,
    out_path:    Path,
) -> None:
    """Write the full event trace to *out_path* as pretty-printed JSON."""

    event_summary = dict(Counter(e.get("type", "unknown") for e in events))

    output = {
        "query":         query,
        "route":         _detect_route(query),
        "server_url":    server_url,
        "started_at":    started_at,
        "finished_at":   finished_at,
        "duration_s":    round(duration_s, 3),
        "event_count":   len(events),
        "event_summary": event_summary,
        "events":        events,
    }

    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> int:
    query      = args.query
    server_url = args.url.rstrip("/") + "/"
    timeout    = args.timeout

    # Build output path
    if args.out:
        out_path = Path(args.out)
    else:
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        route = _detect_route(query)
        slug  = query[:25].replace(" ", "_").replace("/", "-").replace(":", "")
        out_path = Path(f"ag_ui_events_{route}_{ts}_{slug}.json")

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print("━" * 64)
    print(f"  Query   : {query!r}")
    print(f"  Route   : {_detect_route(query)}")
    print(f"  Server  : {server_url}")
    print(f"  Output  : {out_path}")
    print("━" * 64)
    print()

    # ── Collect events ────────────────────────────────────────────────────────
    started_at   = datetime.now(timezone.utc).isoformat()
    t0           = time.perf_counter()

    try:
        events = await collect_events(query, server_url, timeout)
    except Exception as exc:
        print(f"\n  ✗ Error: {exc}", file=sys.stderr)
        return 1

    finished_at = datetime.now(timezone.utc).isoformat()
    duration_s  = time.perf_counter() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("━" * 64)
    print(f"  ✓  {len(events)} events received in {duration_s:.2f}s")
    print()
    print("  Event type breakdown:")
    counts = Counter(e.get("type", "?") for e in events)
    for etype, n in counts.most_common():
        bar = "█" * min(n, 40)
        print(f"    {etype:<38}  {n:>3}  {bar}")
    print("━" * 64)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_to_json(
        events      = events,
        query       = query,
        server_url  = server_url,
        started_at  = started_at,
        finished_at = finished_at,
        duration_s  = duration_s,
        out_path    = out_path,
    )
    print(f"\n  💾  Saved to: {out_path.resolve()}\n")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a query to the MAF MAS server and save all AG-UI events to JSON.",
    )
    parser.add_argument(
        "query",
        help=(
            'Query with routing prefix: '
            '"math: …", "string: …", or "magentic: …"'
        ),
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_SERVER,
        help=f"AG-UI server URL (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="FILE",
        help="Output JSON file path (auto-generated if omitted)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    status = asyncio.run(main(args))
    sys.exit(status)
