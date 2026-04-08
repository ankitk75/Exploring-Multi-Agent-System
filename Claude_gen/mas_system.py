#!/usr/bin/env python3
"""
Multi-Agent System (MAS) — Microsoft Agent Framework (MAF)
===========================================================

Architecture:
┌──────────────────────────────────────────────────────────────────────┐
│                            Chat Layer                                │
│              (keyword router: direct agents vs. magentic)            │
└──────────┬──────────────────────────────────────┬────────────────────┘
           │ complex / multi-step task             │ single-domain task
           ▼                                       ▼
┌───────────────────────────┐        ┌─────────────────────────────┐
│      MagenticManager      │        │  MathAgent  │  StringAgent   │
│    (MagenticBuilder)      │        │  (direct    │  (direct A2A)  │
│  ┌────────────────────┐   │        │   A2A call) │                │
│  │   MathAgent (A2A)  │   │        └─────────────────────────────┘
│  │  MCP: add/mul/div  │   │
│  ├────────────────────┤   │
│  │  StringAgent (A2A) │   │
│  │  MCP: up/lo/rev    │   │
│  └────────────────────┘   │
└───────────────────────────┘

AG-UI Event Streaming (→ Terminal, future: real UI via WebSocket)
─────────────────────────────────────────────────────────────────
  • workflow.run(task, stream=True)    →  yields MAF-native streaming events
  • intermediate_outputs=True         →  surfaces every participant step
  • Custom AG-UI events (CustomEvent) →  needed for A2A inter-agent messages
  • ToolCallStart / ToolCallEnd       →  each MCP tool invocation
  • TextMessageStart / Content / End  →  agent natural-language replies

ANSWER to the key design question
──────────────────────────────────
  Q: Is intermediate_outputs=True + stream=True alone enough to show
     the full inner workings (A2A traffic, tool calls) in a UI?

  A: Partially.
     ✓ intermediate_outputs=True exposes each participant's response
       inside the MagenticBuilder turn-by-turn loop.
     ✓ stream=True lets you consume those events as they arrive.
     ✗ A2A message routing (manager ↔ participant dispatches) is NOT
       automatically emitted as AG-UI events — you must hook the
       MagenticBuilder's on_message callback and emit CustomEvent.
     ✗ Individual tool calls inside participants also need explicit
       on_tool_call / on_tool_result hooks wired to ToolCallStart/End.

  ∴  You need BOTH:
       1. Built-in streaming (intermediate_outputs + stream=True)
       2. Custom AG-UI events for A2A traffic and tool-level visibility.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ▸ Imports & Environment
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import json
import os
import sys
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

# ── Microsoft Agent Framework (MAF) ──────────────────────────────────────────
# pip install agent-framework
from agent_framework.clients import OpenAIChatClient          # Custom-endpoint LLM client
from agent_framework.magentic import MagenticBuilder          # Magentic orchestrator
from agent_framework.agents import Agent                      # Base MAF agent
from agent_framework.tools import Tool                        # MAF tool wrapper
from agent_framework.a2a import A2AAgent, A2AServer           # A2A wrapping

# ── A2A SDK ───────────────────────────────────────────────────────────────────
# pip install a2a-sdk
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
    SendMessageRequest,
    MessageSendParams,
    Message as A2AMessage,
    TextPart,
    Role as A2ARole,
)

# ── MCP (Model Context Protocol) ─────────────────────────────────────────────
# pip install mcp
from mcp.server.fastmcp import FastMCP

# ── AG-UI ─────────────────────────────────────────────────────────────────────
# pip install agent-framework-ag-ui ag-ui-protocol
from agent_framework_ag_ui import AGUIEventEmitter, RunAgentInput
from ag_ui.core import (
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    CustomEvent,
    StateSnapshotEvent,
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ▸ Configuration
# ─────────────────────────────────────────────────────────────────────────────

console = Console()


@dataclass(frozen=True)
class Config:
    """Centralised configuration loaded from the .env file."""
    openai_base_url:  str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_api_key:   str = os.getenv("OPENAI_API_KEY",  "")
    model_name:       str = os.getenv("MODEL_NAME",      "gpt-4o")

    # Logical A2A endpoints (in-process registry resolves these)
    math_agent_url:    str = "http://localhost:8001"
    string_agent_url:  str = "http://localhost:8002"
    manager_agent_url: str = "http://localhost:8000"


CONFIG = Config()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ▸ AG-UI Terminal Renderer
#
# In production replace this class with a WebSocket / SSE emitter that pushes
# events to a real front-end.  Every emit() call here maps 1-to-1 with what
# would be sent over the wire in a production AG-UI setup.
# ─────────────────────────────────────────────────────────────────────────────

class TerminalAGUIRenderer:
    """
    Consumes AG-UI protocol events and renders them to the terminal in real time.

    Supported event types
    ─────────────────────
    RUN_STARTED / RUN_FINISHED     →  run boundary markers
    TEXT_MESSAGE_*                 →  streaming agent text (with agent label)
    TOOL_CALL_START/ARGS/END       →  MCP tool invocations with args + result
    CUSTOM  (a2a_message)          →  A2A inter-agent messages
    CUSTOM  (magentic_step)        →  MagenticBuilder intermediate steps
    CUSTOM  (agent_thinking)       →  agent reasoning traces
    STATE_SNAPSHOT                 →  workflow state dumps
    """

    _COLORS: dict[str, str] = {
        "math":    "bright_yellow",
        "string":  "bright_green",
        "manager": "bright_magenta",
        "a2a":     "cyan",
        "tool":    "bright_blue",
        "system":  "bright_cyan",
        "user":    "white",
        "error":   "bright_red",
    }

    def _agent_color(self, name: str) -> str:
        n = (name or "").lower()
        if "math"    in n: return self._COLORS["math"]
        if "string"  in n: return self._COLORS["string"]
        if "manager" in n or "magentic" in n: return self._COLORS["manager"]
        return "white"

    # ── Public entry point ────────────────────────────────────────────────

    def render(self, event: Any) -> None:
        """Dispatch any AG-UI event to the correct sub-renderer."""
        etype = getattr(event, "type", None)
        dispatch = {
            EventType.RUN_STARTED:           self._run_started,
            EventType.RUN_FINISHED:          self._run_finished,
            EventType.TEXT_MESSAGE_START:    self._text_start,
            EventType.TEXT_MESSAGE_CONTENT:  self._text_content,
            EventType.TEXT_MESSAGE_END:      self._text_end,
            EventType.TOOL_CALL_START:       self._tool_start,
            EventType.TOOL_CALL_ARGS:        self._tool_args,
            EventType.TOOL_CALL_END:         self._tool_end,
            EventType.CUSTOM:                self._custom,
            EventType.STATE_SNAPSHOT:        self._state_snapshot,
        }
        handler = dispatch.get(etype)
        if handler:
            handler(event)
        else:
            console.print(f"[dim]  ⟩ unhandled event type: {etype}[/dim]")

    # ── Handlers ──────────────────────────────────────────────────────────

    def _run_started(self, e: RunStartedEvent) -> None:
        console.print(Rule(
            f"[bold {self._COLORS['system']}]▶  RUN STARTED  "
            f"[dim](run_id={e.run_id})[/dim][/bold {self._COLORS['system']}]"
        ))

    def _run_finished(self, e: RunFinishedEvent) -> None:
        console.print(Rule(
            f"[bold {self._COLORS['system']}]■  RUN FINISHED "
            f"[dim](run_id={e.run_id})[/dim][/bold {self._COLORS['system']}]"
        ))

    def _text_start(self, e: TextMessageStartEvent) -> None:
        agent = getattr(e, "agent_name", None) or getattr(e, "role", "assistant")
        color = self._agent_color(agent)
        console.print(f"\n[{color}]💬  [{agent}][/{color}] ", end="")

    def _text_content(self, e: TextMessageContentEvent) -> None:
        console.print(e.delta, end="", highlight=False)

    def _text_end(self, _e: TextMessageEndEvent) -> None:
        console.print()  # newline after streamed content

    def _tool_start(self, e: ToolCallStartEvent) -> None:
        agent = getattr(e, "agent_name", "")
        tool  = getattr(e, "tool_name",  "unknown_tool")
        color = self._agent_color(agent)
        console.print(
            f"\n  [{color}]🔧  TOOL CALL  [{agent}] → {tool}[/{color}]"
        )

    def _tool_args(self, e: ToolCallArgsEvent) -> None:
        try:
            args = json.loads(e.delta) if isinstance(e.delta, str) else e.delta
            console.print(f"     [dim]args : {json.dumps(args)}[/dim]")
        except (json.JSONDecodeError, TypeError):
            console.print(f"     [dim]args : {e.delta}[/dim]")

    def _tool_end(self, e: ToolCallEndEvent) -> None:
        result = getattr(e, "result", None)
        if result is not None:
            console.print(f"     [{self._COLORS['tool']}]result: {result}[/{self._COLORS['tool']}]")

    def _custom(self, e: CustomEvent) -> None:
        name  = getattr(e, "name",  "")
        value = getattr(e, "value", {})

        if name == "a2a_message":
            # ── A2A inter-agent message ────────────────────────────────────
            src = value.get("from", "?")
            dst = value.get("to",   "?")
            msg = value.get("message", "")
            console.print(Panel(
                f"[cyan]{msg}[/cyan]",
                title=f"[bold cyan]A2A  {src}  →  {dst}[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
                expand=False,
            ))

        elif name == "magentic_step":
            # ── MagenticBuilder intermediate step ─────────────────────────
            step  = value.get("step",        "?")
            agent = value.get("agent",       "")
            desc  = value.get("description", "")
            color = self._agent_color(agent)
            console.print(
                f"  [{color}]⚙  Magentic[{step}]"
                f"{f'  [{agent}]' if agent else ''}  →  {desc}[/{color}]"
            )

        elif name == "agent_thinking":
            # ── Internal reasoning trace ───────────────────────────────────
            agent  = value.get("agent",  "?")
            thought = value.get("thought", "")
            color  = self._agent_color(agent)
            console.print(f"  [{color}]💭  [{agent}] thinking: {thought}[/{color}]")

        else:
            console.print(f"  [dim]• Custom({name}): {value}[/dim]")

    def _state_snapshot(self, e: StateSnapshotEvent) -> None:
        snapshot = getattr(e, "snapshot", {})
        console.print(Panel(
            json.dumps(snapshot, indent=2, default=str),
            title="[bold dim]STATE SNAPSHOT[/bold dim]",
            border_style="dim",
            expand=False,
        ))


# Singleton renderer — all agents share the same emit() function
_RENDERER = TerminalAGUIRenderer()


def emit(event: Any) -> None:
    """
    Global AG-UI event emitter.
    In production: replace the body with a WebSocket/SSE push.
    """
    _RENDERER.render(event)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ▸ MCP Math Server   (add, multiply, divide)
# ─────────────────────────────────────────────────────────────────────────────

math_mcp = FastMCP(
    name="math-mcp-server",
    description="Arithmetic tools for the Math Agent",
)


@math_mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together and return the sum."""
    return a + b


@math_mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b


@math_mcp.tool()
def divide(a: float, b: float) -> float:
    """
    Divide a by b and return the quotient.
    Raises ValueError when b is zero.
    """
    if b == 0:
        raise ValueError("Division by zero is undefined.")
    return a / b


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ▸ MCP String Server  (uppercase, lowercase, reverse)
# ─────────────────────────────────────────────────────────────────────────────

string_mcp = FastMCP(
    name="string-mcp-server",
    description="String transformation tools for the String Agent",
)


@string_mcp.tool()
def to_uppercase(text: str) -> str:
    """Convert every character in the string to uppercase."""
    return text.upper()


@string_mcp.tool()
def to_lowercase(text: str) -> str:
    """Convert every character in the string to lowercase."""
    return text.lower()


@string_mcp.tool()
def reverse_string(text: str) -> str:
    """Return the input string with its characters in reversed order."""
    return text[::-1]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ▸ MCP → MAF Tool Bridge
#
# FastMCP tools must be wrapped into MAF Tool objects so they can be attached
# to MAF Agents.  The bridge calls the underlying Python function directly
# (in-process MCP); for remote MCP servers use mcp.client.stdio / HTTP instead.
# ─────────────────────────────────────────────────────────────────────────────

def _wrap_mcp_as_maf_tool(mcp_server: FastMCP, fn_name: str) -> Tool:
    """
    Wrap a registered FastMCP function as a MAF Tool.

    MAF Tool constructor expects:
        name        : str
        description : str
        fn          : Callable  (sync or async)
        parameters  : dict      (JSON-Schema subset)
    """
    raw_tools: dict[str, Any] = mcp_server._tool_manager._tools  # internal registry

    if fn_name not in raw_tools:
        raise KeyError(f"MCP server '{mcp_server.name}' has no tool '{fn_name}'")

    mcp_tool = raw_tools[fn_name]

    return Tool(
        name=fn_name,
        description=mcp_tool.description or fn_name,
        fn=mcp_tool.fn,
        parameters=mcp_tool.parameters or {},
    )


def load_math_tools() -> list[Tool]:
    return [
        _wrap_mcp_as_maf_tool(math_mcp, "add"),
        _wrap_mcp_as_maf_tool(math_mcp, "multiply"),
        _wrap_mcp_as_maf_tool(math_mcp, "divide"),
    ]


def load_string_tools() -> list[Tool]:
    return [
        _wrap_mcp_as_maf_tool(string_mcp, "to_uppercase"),
        _wrap_mcp_as_maf_tool(string_mcp, "to_lowercase"),
        _wrap_mcp_as_maf_tool(string_mcp, "reverse_string"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 ▸ LLM Client Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_llm_client(system_prompt: str = "") -> OpenAIChatClient:
    """
    Build an OpenAIChatClient pointing at a custom base URL.
    Works with OpenAI, Azure OpenAI, LM Studio, Ollama, etc.
    """
    return OpenAIChatClient(
        model=CONFIG.model_name,
        api_key=CONFIG.openai_api_key,
        base_url=CONFIG.openai_base_url,
        system_prompt=system_prompt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 ▸ AG-UI Tool-Call Hook Mixin
#
# Both MathAgent and StringAgent inherit this mixin so that every MCP tool
# invocation is automatically surfaced as AG-UI ToolCallStart / ToolCallEnd
# events — giving the terminal (and any future UI) per-call visibility.
# ─────────────────────────────────────────────────────────────────────────────

class AGUIToolHooksMixin:
    """
    Mixin that hooks into MAF's on_tool_call / on_tool_result callbacks
    and emits the corresponding AG-UI events.
    """

    # Subclasses must set this attribute.
    AGENT_NAME: str = "UnknownAgent"

    def _on_tool_call(self, tool_name: str, args: dict, call_id: str) -> None:
        emit(ToolCallStartEvent(
            type=EventType.TOOL_CALL_START,
            tool_call_id=call_id,
            tool_name=tool_name,
            agent_name=self.AGENT_NAME,
        ))
        emit(ToolCallArgsEvent(
            type=EventType.TOOL_CALL_ARGS,
            tool_call_id=call_id,
            delta=json.dumps(args),
        ))

    def _on_tool_result(self, tool_name: str, result: Any, call_id: str) -> None:
        emit(ToolCallEndEvent(
            type=EventType.TOOL_CALL_END,
            tool_call_id=call_id,
            result=str(result),
            agent_name=self.AGENT_NAME,
        ))

    def _emit_message(self, text: str) -> None:
        """Emit a complete streamed text message in one shot."""
        msg_id = str(uuid.uuid4())
        emit(TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START,
            message_id=msg_id,
            role="assistant",
            agent_name=self.AGENT_NAME,
        ))
        emit(TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id=msg_id,
            delta=text,
        ))
        emit(TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=msg_id,
        ))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 ▸ Math Agent
# ─────────────────────────────────────────────────────────────────────────────

class MathAgentWrapper(AGUIToolHooksMixin):
    """
    MAF Agent specialised for arithmetic.
    • Tools sourced from MCP math server (add, multiply, divide).
    • Wrapped with A2AAgent for inter-agent communication.
    • All tool calls surfaced as AG-UI events via the mixin.
    """

    AGENT_NAME  = "MathAgent"
    DESCRIPTION = (
        "A precise arithmetic agent.  Handles addition, multiplication, "
        "and division using dedicated MCP tools."
    )
    SYSTEM_PROMPT = (
        "You are MathAgent, a precise arithmetic assistant. "
        "You have three tools: add(a, b), multiply(a, b), divide(a, b). "
        "Use them to solve mathematical questions. "
        "Always explain your reasoning in natural language before giving the answer."
    )

    def __init__(self) -> None:
        self._agent:     Optional[Agent]    = None
        self._a2a_agent: Optional[A2AAgent] = None
        self.card = self._build_card()

    def _build_card(self) -> AgentCard:
        return AgentCard(
            name=self.AGENT_NAME,
            description=self.DESCRIPTION,
            url=CONFIG.math_agent_url,
            version="1.0.0",
            capabilities=AgentCapabilities(streaming=True),
            skills=[
                AgentSkill(id="add",      name="Add",      description="Add two numbers"),
                AgentSkill(id="multiply", name="Multiply", description="Multiply two numbers"),
                AgentSkill(id="divide",   name="Divide",   description="Divide two numbers"),
            ],
        )

    def initialize(self, tools: list[Tool]) -> None:
        """Build the MAF Agent and wrap it with A2AAgent."""
        self._agent = Agent(
            name=self.AGENT_NAME,
            description=self.DESCRIPTION,
            client=make_llm_client(self.SYSTEM_PROMPT),
            tools=tools,
            on_tool_call=self._on_tool_call,
            on_tool_result=self._on_tool_result,
        )
        self._a2a_agent = A2AAgent(
            agent=self._agent,
            card=self.card,
        )
        console.print(
            f"[{_RENDERER._COLORS['math']}]"
            f"✓  MathAgent initialised with {len(tools)} MCP tools[/]"
        )

    async def run(self, message: str) -> str:
        """Execute a task, emitting AG-UI events throughout."""
        self._emit_message_start()
        response: str = await self._agent.run(message)
        self._emit_message(response)
        return response

    def _emit_message_start(self) -> None:
        pass  # _emit_message() handles start+content+end atomically

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def a2a_agent(self) -> A2AAgent:
        return self._a2a_agent


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 ▸ String Agent
# ─────────────────────────────────────────────────────────────────────────────

class StringAgentWrapper(AGUIToolHooksMixin):
    """
    MAF Agent specialised for string transformations.
    • Tools sourced from MCP string server (upper, lower, reverse).
    • Wrapped with A2AAgent for inter-agent communication.
    """

    AGENT_NAME  = "StringAgent"
    DESCRIPTION = (
        "A string-transformation agent.  Handles uppercase conversion, "
        "lowercase conversion, and string reversal via MCP tools."
    )
    SYSTEM_PROMPT = (
        "You are StringAgent, a string-manipulation assistant. "
        "You have three tools: to_uppercase(text), to_lowercase(text), "
        "and reverse_string(text). "
        "Use them to transform text as requested. "
        "Always confirm what you did in natural language."
    )

    def __init__(self) -> None:
        self._agent:     Optional[Agent]    = None
        self._a2a_agent: Optional[A2AAgent] = None
        self.card = self._build_card()

    def _build_card(self) -> AgentCard:
        return AgentCard(
            name=self.AGENT_NAME,
            description=self.DESCRIPTION,
            url=CONFIG.string_agent_url,
            version="1.0.0",
            capabilities=AgentCapabilities(streaming=True),
            skills=[
                AgentSkill(id="uppercase", name="Uppercase", description="Convert text to uppercase"),
                AgentSkill(id="lowercase", name="Lowercase", description="Convert text to lowercase"),
                AgentSkill(id="reverse",   name="Reverse",   description="Reverse a string"),
            ],
        )

    def initialize(self, tools: list[Tool]) -> None:
        self._agent = Agent(
            name=self.AGENT_NAME,
            description=self.DESCRIPTION,
            client=make_llm_client(self.SYSTEM_PROMPT),
            tools=tools,
            on_tool_call=self._on_tool_call,
            on_tool_result=self._on_tool_result,
        )
        self._a2a_agent = A2AAgent(
            agent=self._agent,
            card=self.card,
        )
        console.print(
            f"[{_RENDERER._COLORS['string']}]"
            f"✓  StringAgent initialised with {len(tools)} MCP tools[/]"
        )

    async def run(self, message: str) -> str:
        response: str = await self._agent.run(message)
        self._emit_message(response)
        return response

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def a2a_agent(self) -> A2AAgent:
        return self._a2a_agent


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 ▸ A2A Registry & Card Resolver
#
# In production each agent is a separately deployed HTTP service and
# A2ACardResolver fetches Agent Cards over the network.
# Here we keep everything in-process: registration and resolution are local.
# ─────────────────────────────────────────────────────────────────────────────

class InProcessA2ARegistry:
    """
    Lightweight in-process A2A registry.

    Roles:
      • Stores agent cards and A2AAgent references keyed by name.
      • Provides send_message() for A2A inter-agent communication,
        emitting CustomEvent("a2a_message") for every dispatch so the
        terminal (and any future UI) can show the traffic in real time.
      • Exposes an A2ACardResolver for agent discovery.
    """

    def __init__(self) -> None:
        self._a2a_agents: dict[str, A2AAgent] = {}
        self._cards:      dict[str, AgentCard] = {}
        self.resolver = A2ACardResolver(base_url="http://localhost")

    def register(self, name: str, a2a_agent: A2AAgent, card: AgentCard) -> None:
        self._a2a_agents[name] = a2a_agent
        self._cards[name]      = card
        console.print(
            f"  [cyan]A2A registry → registered '{name}' "
            f"at {card.url}[/cyan]"
        )

    def get_card(self, name: str) -> Optional[AgentCard]:
        return self._cards.get(name)

    def list_agents(self) -> list[str]:
        return list(self._a2a_agents.keys())

    async def send_message(
        self,
        from_agent: str,
        to_agent:   str,
        message:    str,
    ) -> str:
        """
        Route a message from one agent to another via A2A.

        Why CustomEvent is required here
        ─────────────────────────────────
        MagenticBuilder's built-in stream does NOT automatically emit
        AG-UI events for cross-agent dispatches.  We must fire a
        CustomEvent("a2a_message") ourselves so the UI can observe
        every manager ↔ participant exchange.
        """
        # 1. Emit AG-UI custom event for terminal / UI visibility
        emit(CustomEvent(
            type=EventType.CUSTOM,
            name="a2a_message",
            value={
                "from":    from_agent,
                "to":      to_agent,
                "message": message,
            },
        ))

        # 2. Resolve target agent and dispatch
        target_wrapper = self._a2a_agents.get(to_agent)
        if target_wrapper is None:
            raise ValueError(
                f"A2A Registry: unknown agent '{to_agent}'. "
                f"Registered: {self.list_agents()}"
            )

        response: str = await target_wrapper.agent.run(message)
        return response


# Module-level singleton registry
A2A_REGISTRY = InProcessA2ARegistry()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 ▸ Magentic Manager  (MagenticBuilder)
# ─────────────────────────────────────────────────────────────────────────────

class MagenticManagerWrapper:
    """
    Wraps MAF's MagenticBuilder.

    Key configuration
    ─────────────────
    • intermediate_outputs=True   surfaces every participant response step
    • on_message callback         fires for each A2A dispatch, letting us
                                  emit CustomEvent("a2a_message") events
    • workflow.run(task, stream=True)
                                  returns an AsyncGenerator of MAF events;
                                  we adapt each one to AG-UI in _adapt_event()

    Why custom AG-UI events are ALSO needed
    ────────────────────────────────────────
    Even with intermediate_outputs=True, MAF emits its own internal event
    objects — not AG-UI protocol events.  _adapt_event() bridges the two.
    Additionally, A2A routing traffic (manager ↔ participants) is not part
    of MAF's built-in event stream; the on_message hook + CustomEvent fills
    that gap.
    """

    NAME = "MagenticManager"

    def __init__(self, math_agent: Agent, string_agent: Agent) -> None:
        self._workflow = MagenticBuilder(
            participants=[math_agent, string_agent],
            client=make_llm_client(),
            intermediate_outputs=True,       # ← expose per-participant steps
            on_message=self._on_a2a_dispatch, # ← capture A2A routing traffic
        )
        console.print(
            f"[{_RENDERER._COLORS['manager']}]"
            f"✓  MagenticManager initialised "
            f"(participants: MathAgent, StringAgent)[/]"
        )

    # ── A2A dispatch hook ─────────────────────────────────────────────────

    def _on_a2a_dispatch(
        self,
        from_agent: str,
        to_agent:   str,
        message:    str,
    ) -> None:
        """
        Called by MagenticBuilder every time it routes a sub-task to a
        participant.  We emit a CustomEvent so the terminal shows the
        A2A traffic in real time.
        """
        emit(CustomEvent(
            type=EventType.CUSTOM,
            name="a2a_message",
            value={
                "from":    from_agent or self.NAME,
                "to":      to_agent,
                "message": message,
            },
        ))

    # ── Main run loop ─────────────────────────────────────────────────────

    async def run(self, task: str) -> AsyncGenerator[Any, None]:
        """
        Execute the magentic workflow with event streaming.

        Flow:
          1. Emit RunStartedEvent
          2. Emit magentic_step(start) custom event
          3. Stream MAF events → adapt each to AG-UI → emit + yield
          4. Emit RunFinishedEvent
        """
        run_id    = str(uuid.uuid4())
        thread_id = str(uuid.uuid4())

        emit(RunStartedEvent(
            type=EventType.RUN_STARTED,
            run_id=run_id,
            thread_id=thread_id,
        ))
        emit(CustomEvent(
            type=EventType.CUSTOM,
            name="magentic_step",
            value={"step": "start", "description": f"Task: {task}"},
        ))

        async for maf_event in self._workflow.run(task, stream=True):
            ag_ui_event = self._adapt_event(maf_event, run_id)
            emit(ag_ui_event)
            yield ag_ui_event

        emit(RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            run_id=run_id,
        ))

    # ── MAF → AG-UI event adapter ─────────────────────────────────────────

    def _adapt_event(self, maf_event: Any, run_id: str) -> Any:
        """
        Translate a MAF-native streaming event into an AG-UI event.

        MAF event shape (approximate):
            .type   : str   e.g. "text_delta", "tool_call", "tool_result",
                                 "intermediate_output", "final_output"
            .data   : dict  payload varies by type
        """
        etype      = getattr(maf_event, "type", "unknown")
        data       = getattr(maf_event, "data", {})
        data_dict  = data if isinstance(data, dict) else {}
        agent_name = data_dict.get("agent", self.NAME)
        msg_id     = str(uuid.uuid4())
        call_id    = data_dict.get("call_id", str(uuid.uuid4()))

        if etype == "text_delta":
            return TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=msg_id,
                delta=data_dict.get("text", str(data)),
                agent_name=agent_name,
            )

        elif etype == "tool_call":
            # Emit start + args as side-effects; return start
            args_event = ToolCallArgsEvent(
                type=EventType.TOOL_CALL_ARGS,
                tool_call_id=call_id,
                delta=json.dumps(data_dict.get("args", {})),
            )
            emit(args_event)
            return ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id=call_id,
                tool_name=data_dict.get("name", "unknown"),
                agent_name=agent_name,
            )

        elif etype == "tool_result":
            return ToolCallEndEvent(
                type=EventType.TOOL_CALL_END,
                tool_call_id=call_id,
                result=str(data_dict.get("result", data)),
                agent_name=agent_name,
            )

        elif etype in ("intermediate_output", "participant_output"):
            return CustomEvent(
                type=EventType.CUSTOM,
                name="magentic_step",
                value={
                    "step":        "intermediate",
                    "agent":       agent_name,
                    "description": str(data),
                },
            )

        elif etype == "final_output":
            return CustomEvent(
                type=EventType.CUSTOM,
                name="magentic_step",
                value={
                    "step":        "final",
                    "agent":       agent_name,
                    "description": str(data),
                },
            )

        else:
            # Pass-through unknown events as generic custom events
            return CustomEvent(
                type=EventType.CUSTOM,
                name=f"maf_{etype}",
                value={"data": str(data), "agent": agent_name},
            )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 ▸ Chat Layer  (Router)
#
# This layer sits between the user and the agent system.  Its job is to
# inspect the incoming query and decide which execution path to take:
#
#   ① Direct route to MathAgent   (keyword: purely arithmetic)
#   ② Direct route to StringAgent (keyword: purely string-related)
#   ③ Magentic workflow            (multi-step / ambiguous / combined)
#
# The current router is keyword-based for clarity.  In production, swap
# _route() with a lightweight classifier call or a rules engine without
# touching any agent logic — this is the decoupling benefit of the chat layer.
# ─────────────────────────────────────────────────────────────────────────────

class ChatLayer:
    """
    User-facing entry point that routes queries to the appropriate back-end.
    """

    _MATH_KEYWORDS: frozenset[str] = frozenset({
        "add", "plus", "sum", "multiply", "times", "product",
        "divide", "quotient", "division", "arithmetic", "calculate",
        "+", "*", "/",
    })
    _STRING_KEYWORDS: frozenset[str] = frozenset({
        "uppercase", "lowercase", "reverse", "upper", "lower",
        "string", "text", "characters", "flip", "invert",
    })

    def __init__(
        self,
        manager:       MagenticManagerWrapper,
        math_wrapper:  MathAgentWrapper,
        string_wrapper: StringAgentWrapper,
    ) -> None:
        self._manager = manager
        self._math    = math_wrapper
        self._string  = string_wrapper

    # ── Router ────────────────────────────────────────────────────────────

    def _route(self, query: str) -> str:
        """
        Returns one of: "math_direct" | "string_direct" | "magentic"
        """
        tokens = frozenset(query.lower().split())
        has_math   = bool(tokens & self._MATH_KEYWORDS)
        has_string = bool(tokens & self._STRING_KEYWORDS)

        if has_math and not has_string:
            return "math_direct"
        if has_string and not has_math:
            return "string_direct"
        return "magentic"

    # ── Main handler ──────────────────────────────────────────────────────

    async def handle(self, query: str) -> str:
        """Process one user query end-to-end."""
        route = self._route(query)

        console.print(Rule("[bold white]USER QUERY[/bold white]"))
        console.print(Panel(query, border_style="white", expand=False))
        console.print(
            f"\n  [dim]Router decision:[/dim] [bold]{route}[/bold]\n"
        )

        if route == "math_direct":
            console.print(
                f"  [{_RENDERER._COLORS['math']}]"
                f"↳  Direct route → MathAgent (via A2A)[/]"
            )
            # Simulate A2A dispatch from chat layer → MathAgent
            return await A2A_REGISTRY.send_message(
                from_agent="ChatLayer",
                to_agent="MathAgent",
                message=query,
            )

        elif route == "string_direct":
            console.print(
                f"  [{_RENDERER._COLORS['string']}]"
                f"↳  Direct route → StringAgent (via A2A)[/]"
            )
            return await A2A_REGISTRY.send_message(
                from_agent="ChatLayer",
                to_agent="StringAgent",
                message=query,
            )

        else:  # magentic
            console.print(
                f"  [{_RENDERER._COLORS['manager']}]"
                f"↳  Magentic workflow → MagenticManager[/]\n"
            )
            async for _event in self._manager.run(query):
                pass  # events are rendered by emit() inside run()
            return ""  # final answer already printed via event stream


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 ▸ System Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

async def bootstrap() -> ChatLayer:
    """
    Initialise all system components in the correct dependency order.

    Order:
      1. Load MCP tools (math + string)
      2. Initialise MAF agents + A2A wrappers
      3. Register agents in A2A registry
      4. Create Magentic Manager
      5. Create Chat Layer
    """
    console.print(Rule(
        f"[bold {_RENDERER._COLORS['system']}]  MAS BOOTSTRAP  [/bold {_RENDERER._COLORS['system']}]"
    ))

    # ── 1. MCP tools ──────────────────────────────────────────────────────
    console.print("\n[dim]Loading MCP tools …[/dim]")
    math_tools   = load_math_tools()
    string_tools = load_string_tools()
    console.print(
        f"  [dim]Math tools  : {[t.name for t in math_tools]}[/dim]\n"
        f"  [dim]String tools: {[t.name for t in string_tools]}[/dim]"
    )

    # ── 2. Agents ─────────────────────────────────────────────────────────
    console.print("\n[dim]Initialising agents …[/dim]")
    math_wrapper   = MathAgentWrapper()
    string_wrapper = StringAgentWrapper()

    math_wrapper.initialize(math_tools)
    string_wrapper.initialize(string_tools)

    # ── 3. A2A registry ───────────────────────────────────────────────────
    console.print("\n[dim]Registering agents in A2A registry …[/dim]")
    A2A_REGISTRY.register("MathAgent",   math_wrapper.a2a_agent,   math_wrapper.card)
    A2A_REGISTRY.register("StringAgent", string_wrapper.a2a_agent, string_wrapper.card)

    # ── 4. Magentic Manager ───────────────────────────────────────────────
    console.print("\n[dim]Building Magentic Manager …[/dim]")
    manager = MagenticManagerWrapper(
        math_agent=math_wrapper.agent,
        string_agent=string_wrapper.agent,
    )

    # ── 5. Chat Layer ─────────────────────────────────────────────────────
    chat = ChatLayer(manager, math_wrapper, string_wrapper)

    console.print(Rule(
        f"[bold {_RENDERER._COLORS['system']}]  SYSTEM READY  [/bold {_RENDERER._COLORS['system']}]"
    ))
    console.print()
    return chat


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 ▸ Demo Queries
# ─────────────────────────────────────────────────────────────────────────────

DEMO_QUERIES: list[str] = [
    # ── Direct → MathAgent ────────────────────────────────────────────────
    "What is 42 multiplied by 7?",
    "Divide 100 by 4 and add 5 to the result.",
    # ── Direct → StringAgent ──────────────────────────────────────────────
    "Can you reverse the string 'Hello World'?",
    "Convert 'Python Agent Framework' to uppercase.",
    # ── Magentic (multi-step / combined) ──────────────────────────────────
    "First multiply 6 by 9, then reverse the string representation of the result.",
    "Convert 'MAGENTIC' to lowercase, then also compute 128 divided by 8.",
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 ▸ Interactive REPL
# ─────────────────────────────────────────────────────────────────────────────

async def interactive_loop(chat: ChatLayer) -> None:
    """Simple read-eval-print loop for manual testing."""
    console.print(
        "[dim]Commands: type a query, 'demo' to run demo queries, "
        "'agents' to list registered agents, 'exit' to quit.[/dim]\n"
    )

    while True:
        try:
            query = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Shutting down.[/dim]")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            console.print("[dim]Shutting down.[/dim]")
            break

        if query.lower() == "demo":
            for q in DEMO_QUERIES:
                await chat.handle(q)
                console.print()
            continue

        if query.lower() == "agents":
            console.print(
                f"Registered A2A agents: {A2A_REGISTRY.list_agents()}"
            )
            continue

        await chat.handle(query)
        console.print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17 ▸ Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    # Validate required environment variables
    if not CONFIG.openai_api_key:
        console.print(
            f"[{_RENDERER._COLORS['error']}]"
            f"ERROR: OPENAI_API_KEY is not set.  "
            f"Please check your .env file.[/]"
        )
        sys.exit(1)

    console.print(Panel(
        "[bold]Multi-Agent System[/bold]\n"
        "Microsoft Agent Framework · MCP · A2A · AG-UI\n\n"
        f"  Model  : [cyan]{CONFIG.model_name}[/cyan]\n"
        f"  Endpoint: [cyan]{CONFIG.openai_base_url}[/cyan]",
        border_style="bright_cyan",
        expand=False,
    ))

    chat = await bootstrap()
    await interactive_loop(chat)


if __name__ == "__main__":
    asyncio.run(main())
