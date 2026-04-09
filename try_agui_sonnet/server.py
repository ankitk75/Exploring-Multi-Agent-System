"""
server.py  ·  MAF Multi-Agent System with AG-UI integration
============================================================

Routing logic (based on the first word of the user message):
  "math: <query>"     → MathAgent   (tools: add · multiply · divide)
  "string: <query>"   → StringAgent (tools: to_uppercase · to_lowercase · reverse_text)
  "magentic: <query>" → Magentic orchestration (manager + MathAgent + StringAgent)

The MathAgent and StringAgent are reused as standalone specialists *and* as
Magentic participants, exactly as described in the MAF + AG-UI architecture guide.

Usage
-----
    cp .env.example .env        # fill in your credentials
    python server.py            # starts on http://0.0.0.0:8888
    python client.py "math: what is (17 * 4) + 9?"
    python client.py "string: reverse 'Hello World' then uppercase it"
    python client.py "magentic: compute 8 * 9 and reverse the resulting digits"

Architecture
------------
                          ┌──────────────────────────────────┐
  AG-UI POST /  ──────►  │  RouterWorkflow.run()            │
                          │                                  │
                          │  detect_route()                  │
                          │  strip_prefix()                  │
                          │        │                         │
                          │   ┌────┴──────────────────┐      │
                          │   │math │ string │magentic │      │
                          │   ▼     ▼        ▼         │      │
                          │  math_wf str_wf  mag_wf    │      │
                          │  (WFB)  (WFB)   (MagB)     │      │
                          │        yield WorkflowEvents │      │
                          └──────────────────────────────┘
                                        │
                          AG-UI SSE bridge (agent_framework_ag_ui)
                                        │
                          SSE stream → client.py → events.json

Notes on MAF API versions
--------------------------
This code targets Microsoft Agent Framework ≥ 1.0.0.
If your version differs slightly, the comments marked "[API NOTE]" indicate
the most likely alternative spellings.
"""

from __future__ import annotations

import os
from typing import Annotated

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import Field

# ── MAF core ─────────────────────────────────────────────────────────────────
from agent_framework import Agent, ChatMessage, WorkflowBuilder, tool
from agent_framework.openai import OpenAIChatCompletionClient

# ── MAF Magentic orchestration ───────────────────────────────────────────────
# [API NOTE] If MagenticBuilder lives in agent_framework directly, change to:
#   from agent_framework import MagenticBuilder
from agent_framework_orchestrations import MagenticBuilder

# ── MAF AG-UI bridge ─────────────────────────────────────────────────────────
# [API NOTE] In some MAF versions this is under agent_framework.ag_ui:
#   from agent_framework.ag_ui import add_agent_framework_fastapi_endpoint
from agent_framework_ag_ui import add_agent_framework_fastapi_endpoint

load_dotenv()


# ═════════════════════════════════════════════════════════════════════════════
# 1 ·  LLM CLIENT FACTORY
# ═════════════════════════════════════════════════════════════════════════════

def make_client() -> OpenAIChatCompletionClient:
    """
    Build an OpenAI-compatible chat client from environment variables.

    Required env vars:
        OPENAI_API_KEY   – your API key
        OPENAI_MODEL     – e.g. "gpt-4o" or any deployed model name

    Optional env vars:
        OPENAI_BASE_URL  – custom endpoint (Azure, proxy, local server …).
                           Leave unset to use the default api.openai.com.
    """
    return OpenAIChatCompletionClient(
        model    = os.environ["OPENAI_MODEL"],
        api_key  = os.environ["OPENAI_API_KEY"],
        base_url = os.environ.get("OPENAI_BASE_URL"),   # None → OpenAI default
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2 ·  TOOLS
# ═════════════════════════════════════════════════════════════════════════════

# ── Math tools ────────────────────────────────────────────────────────────────

@tool
def add(
    a: Annotated[float, Field(description="First addend")],
    b: Annotated[float, Field(description="Second addend")],
) -> float:
    """Add two numbers and return the sum."""
    return a + b


@tool
def multiply(
    a: Annotated[float, Field(description="First factor")],
    b: Annotated[float, Field(description="Second factor")],
) -> float:
    """Multiply two numbers and return the product."""
    return a * b


@tool
def divide(
    a: Annotated[float, Field(description="Dividend")],
    b: Annotated[float, Field(description="Divisor – must not be zero")],
) -> float:
    """Divide a by b.  Raises ValueError if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


# ── String tools ──────────────────────────────────────────────────────────────

@tool
def to_uppercase(
    text: Annotated[str, Field(description="Input text to transform")],
) -> str:
    """Return the input string converted to UPPER CASE."""
    return text.upper()


@tool
def to_lowercase(
    text: Annotated[str, Field(description="Input text to transform")],
) -> str:
    """Return the input string converted to lower case."""
    return text.lower()


@tool
def reverse_text(
    text: Annotated[str, Field(description="Input text to reverse")],
) -> str:
    """Return the input string with its characters in reverse order."""
    return text[::-1]


# ═════════════════════════════════════════════════════════════════════════════
# 3 ·  AGENT FACTORIES
#      Each factory is called once at startup; the resulting Agent instance is
#      reused across requests AND passed to MagenticBuilder as a participant.
# ═════════════════════════════════════════════════════════════════════════════

def make_math_agent(client: OpenAIChatCompletionClient) -> Agent:
    return Agent(
        client       = client,
        name         = "MathAgent",
        instructions = (
            "You are a meticulous math assistant. "
            "You MUST use the provided tools (add, multiply, divide) for every "
            "numerical operation — never perform arithmetic in your head. "
            "Present each calculation step and its tool result clearly before "
            "giving the final answer."
        ),
        tools = [add, multiply, divide],
    )


def make_string_agent(client: OpenAIChatCompletionClient) -> Agent:
    return Agent(
        client       = client,
        name         = "StringAgent",
        instructions = (
            "You are a text-transformation specialist. "
            "Use the provided tools (to_uppercase, to_lowercase, reverse_text) "
            "to manipulate strings exactly as requested. "
            "Chain tool calls when multiple transformations are needed."
        ),
        tools = [to_uppercase, to_lowercase, reverse_text],
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4 ·  ROUTING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

# Map lowercase prefix → internal route key
_PREFIX_TO_ROUTE: dict[str, str] = {
    "math:":     "math",
    "string:":   "string",
    "magentic:": "magentic",
}


def detect_route(messages: list[ChatMessage]) -> str:
    """
    Scan messages from newest to oldest; return the route key of the first
    user message that starts with a known prefix.
    Falls back to "math" if no prefix is found.
    """
    for msg in reversed(messages):
        if msg.role == "user":
            text = (msg.content or "").strip().lower()
            for prefix, route in _PREFIX_TO_ROUTE.items():
                if text.startswith(prefix):
                    return route
    return "math"


def strip_prefix(messages: list[ChatMessage]) -> list[ChatMessage]:
    """
    Return a copy of *messages* where the routing prefix has been removed
    from the most recent user turn so the downstream agent receives only
    the actual query.
    """
    # Shallow-copy every message; we only mutate the last user message.
    result = [ChatMessage(role=m.role, content=m.content) for m in messages]

    for i in range(len(result) - 1, -1, -1):
        if result[i].role == "user":
            raw = (result[i].content or "").strip()
            for prefix in _PREFIX_TO_ROUTE:
                if raw.lower().startswith(prefix):
                    result[i] = ChatMessage(
                        role    = "user",
                        content = raw[len(prefix):].strip(),
                    )
            break   # only touch the last user turn

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 5 ·  ROUTER WORKFLOW
#      Implements AgentProtocol via duck-typing:
#      MAF's AG-UI integration calls `agent.run(messages, stream=True)`.
# ═════════════════════════════════════════════════════════════════════════════

class RouterWorkflow:
    """
    Master router that acts as the single AG-UI entry-point for the MAS.

    On each request it:
      1. Detects which route the user prefix specifies.
      2. Strips that prefix from the message list.
      3. Delegates to the matching sub-workflow.
      4. Yields every WorkflowEvent from the sub-workflow unchanged, so the
         AG-UI SSE bridge receives the full inner-working event stream
         (MagenticOrchestratorEvent, MagenticProgressLedger, TOOL_CALL_*,
         TEXT_MESSAGE_*, STEP_*, etc.).

    Thread-safety note
    ------------------
    A single RouterWorkflow instance is shared across all HTTP requests.
    The MAF workflow objects (`_math_wf`, `_string_wf`, `_magentic_wf`) are
    built from stateless agent instances, so concurrent requests are safe as
    long as each `.run()` call is independent (which is the MAF contract).

    For strict per-request isolation (e.g. if you add stateful middleware),
    move the RouterWorkflow instantiation inside a `workflow_factory` lambda
    and wrap with `AgentFrameworkWorkflow`:

        from agent_framework_ag_ui import AgentFrameworkWorkflow
        agui = AgentFrameworkWorkflow(
            workflow_factory = lambda _tid: RouterWorkflow(),
            name             = "mas_router",
        )
        add_agent_framework_fastapi_endpoint(app, agui, "/")
    """

    def __init__(self) -> None:
        # One shared LLM client for the specialist agents.
        # The Magentic manager gets its own client to keep usage separate.
        agent_client   = make_client()
        manager_client = make_client()

        # ── Specialist agents (reused in standalone workflows AND as Magentic
        #    participants — this is the key architectural design).
        math_agent   = make_math_agent(agent_client)
        string_agent = make_string_agent(agent_client)

        # ── Single-agent workflows
        #    Wrapping each Agent in WorkflowBuilder gives us a uniform
        #    WorkflowEvent stream across all three routing branches.
        #    [API NOTE] WorkflowBuilder may be chained or imperative depending
        #    on your MAF version.  Both styles shown:

        # Imperative style (MAF ≥ 1.0):
        math_builder = WorkflowBuilder()
        math_builder.set_start_executor(math_agent)
        self._math_wf = math_builder.build()

        string_builder = WorkflowBuilder()
        string_builder.set_start_executor(string_agent)
        self._string_wf = string_builder.build()

        # ── Magentic multi-agent workflow
        #    [API NOTE] MagenticBuilder.with_manager() accepts a chat client
        #    (MAF ≥ 1.0) or an Agent (some earlier previews).  If the call
        #    fails, try: .with_manager(manager_agent) where
        #    manager_agent = Agent(client=manager_client, name="MagenticManager",
        #                          instructions="You are a task orchestrator …")
        self._magentic_wf = (
            MagenticBuilder()
            .with_manager(manager_client)
            .with_participants([math_agent, string_agent])
            .build()
        )

        # Keep a human-readable map for logging.
        self._route_map = {
            "math":     ("MathAgent",    self._math_wf),
            "string":   ("StringAgent",  self._string_wf),
            "magentic": ("Magentic-MAS", self._magentic_wf),
        }

    # ── AgentProtocol duck-typing ─────────────────────────────────────────────

    async def run(
        self,
        messages: list[ChatMessage],
        *,
        stream: bool = True,
        **kwargs,
    ):
        """
        Route the request and stream every inner WorkflowEvent back to the
        caller (the AG-UI SSE bridge).

        The `yield` statement makes this an async generator; Python / MAF
        treat it as such regardless of the return annotation.
        """
        route          = detect_route(messages)
        clean_messages = strip_prefix(messages)
        label, wf      = self._route_map[route]

        print(f"[Router] → {label}  |  query: {clean_messages[-1].content!r:.80}")

        # Pass through every event from the target workflow unchanged.
        # For specialist agents this includes standard WorkflowEvents wrapping
        # AgentResponseUpdate / tool-call payloads.
        # For Magentic this also includes MagenticOrchestratorEvent,
        # MagenticProgressLedger, GroupChatRequestSentEvent, etc.
        async for event in wf.run(messages=clean_messages, stream=True):
            yield event


# ═════════════════════════════════════════════════════════════════════════════
# 6 ·  FASTAPI APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "MAF Multi-Agent System",
    description = "Routes to MathAgent, StringAgent, or Magentic orchestration via AG-UI.",
    version     = "1.0.0",
)

# Single router instance – stateless per-run, so concurrent requests are safe.
_router = RouterWorkflow()

# Register the single AG-UI endpoint.
# The integration:
#   • Accepts POST /  with a JSON body matching RunAgentInput (thread_id,
#     run_id, messages, state).
#   • Calls _router.run(messages, stream=True).
#   • Converts every yielded WorkflowEvent to an AG-UI SSE event
#     (RUN_STARTED, TEXT_MESSAGE_*, TOOL_CALL_*, magentic_orchestrator …).
#   • Streams those events back as Server-Sent Events.
add_agent_framework_fastapi_endpoint(
    app   = app,
    agent = _router,   # AgentProtocol-compatible via duck-typing
    path  = "/",
)


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "routes": list(_router._route_map.keys())}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("━" * 60)
    print("  MAF Multi-Agent System  ·  http://0.0.0.0:8888")
    print("━" * 60)
    print("  Routing prefixes:")
    print('    math: <query>      → MathAgent')
    print('    string: <query>    → StringAgent')
    print('    magentic: <query>  → Magentic orchestration')
    print("━" * 60)
    uvicorn.run(
        "server:app",
        host      = "0.0.0.0",
        port      = 8888,
        reload    = False,
        log_level = "info",
    )
