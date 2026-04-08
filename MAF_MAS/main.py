"""
=============================================================================
 Multi-Agent System (MAS) — Microsoft Agent Framework
=============================================================================
 Architecture:
   • Magentic Manager  — orchestrates the workflow (MagenticBuilder)
   • Math Agent        — tools via MCP (add, multiply, divide)
   • String Agent      — tools via MCP (uppercase, lowercase, reverse)

 Protocols:
   • MCP   — tool integration (stdio servers)
   • A2A   — agent-to-agent communication wrappers
   • AG-UI — real-time event streaming to UI (terminal in this demo)

 Observability:
   • intermediate_outputs=True → surfaces all inner agent exchange
   • workflow.run(task, stream=True) → streams every event in real time
   • Custom event logging for tool calls and A2A activity
=============================================================================
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Agent Framework Core ─────────────────────────────────────────────────────
from agent_framework import (
    Agent,
    AgentResponseUpdate,
    MCPStdioTool,
    Message,
    WorkflowEvent,
)

# ── OpenAI Chat Client (with custom endpoint support) ────────────────────────
from agent_framework.openai import OpenAIChatClient

# ── Orchestrations (MagenticBuilder) ─────────────────────────────────────────
from agent_framework.orchestrations import (
    GroupChatRequestSentEvent,
    MagenticBuilder,
    MagenticProgressLedger,
)

# ── A2A Protocol (agent-to-agent communication) ─────────────────────────────
from a2a.client import A2ACardResolver  # noqa: F401 — imported per requirements
from agent_framework.a2a import A2AAgent  # noqa: F401 — imported per requirements

# ── AG-UI Protocol (UI event streaming) ──────────────────────────────────────
import agent_framework_ag_ui  # noqa: F401 — imported per requirements


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

load_dotenv()

# Resolve the directory where MCP server scripts live (same dir as this file)
BASE_DIR = Path(__file__).resolve().parent

# OpenAI-compatible endpoint configuration
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")

if not OPENAI_API_KEY:
    print("\n[ERROR] OPENAI_API_KEY is not set. Please configure it in .env")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: TERMINAL UI HELPERS (AG-UI Event Rendering)
# ═══════════════════════════════════════════════════════════════════════════════

# ANSI colour codes for rich terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    WHITE = "\033[97m"
    BG_DARK = "\033[48;5;235m"
    UNDERLINE = "\033[4m"


def print_header(text: str) -> None:
    """Print a styled section header."""
    width = 80
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}{Colors.RESET}\n")


def print_event(label: str, detail: str, color: str = Colors.DIM) -> None:
    """Print a formatted event line."""
    print(f"  {color}▸ {Colors.BOLD}{label}{Colors.RESET}{color}  {detail}{Colors.RESET}")


def print_agent_token(executor_id: str, text: str, is_new: bool = False) -> None:
    """Print streaming token output from an agent."""
    if is_new:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}🤖 [{executor_id}]:{Colors.RESET} ", end="", flush=True)
    print(f"{Colors.WHITE}{text}{Colors.RESET}", end="", flush=True)


def print_tool_call(agent_name: str, tool_name: str, args: str) -> None:
    """Print a tool call event."""
    print(f"  {Colors.YELLOW}🔧 [{agent_name}] Tool Call → {Colors.BOLD}{tool_name}{Colors.RESET}"
          f"{Colors.YELLOW}({args}){Colors.RESET}")


def print_a2a_event(source: str, target: str, message: str) -> None:
    """Print an A2A communication event."""
    print(f"  {Colors.MAGENTA}🔗 [A2A] {source} → {target}: {message}{Colors.RESET}")


def print_orchestrator_event(event_type: str, detail: str) -> None:
    """Print a Magentic orchestrator event."""
    print(f"  {Colors.BLUE}📋 [Orchestrator] {Colors.BOLD}{event_type}{Colors.RESET}"
          f"{Colors.BLUE}: {detail}{Colors.RESET}")


def print_separator() -> None:
    """Print a thin separator line."""
    print(f"  {Colors.DIM}{'─' * 76}{Colors.RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: OPENAI CHAT CLIENT SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def create_chat_client() -> OpenAIChatClient:
    """
    Create an OpenAIChatClient pointed at the configured endpoint.
    Supports any OpenAI-compatible API (OpenAI, Azure, Ollama, etc.).
    """
    return OpenAIChatClient(
        model=OPENAI_MODEL,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MCP TOOL SERVER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_math_mcp_tool() -> MCPStdioTool:
    """
    Create an MCPStdioTool that launches the math MCP server as a subprocess.
    Tools exposed: add, multiply, divide
    """
    return MCPStdioTool(
        name="MathTools",
        command=sys.executable,  # Use the current Python interpreter
        args=[str(BASE_DIR / "math_server.py")],
    )


def create_string_mcp_tool() -> MCPStdioTool:
    """
    Create an MCPStdioTool that launches the string MCP server as a subprocess.
    Tools exposed: to_uppercase, to_lowercase, reverse
    """
    return MCPStdioTool(
        name="StringTools",
        command=sys.executable,  # Use the current Python interpreter
        args=[str(BASE_DIR / "string_server.py")],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: AGENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_math_agent(client: OpenAIChatClient, mcp_tool: MCPStdioTool) -> Agent:
    """
    Create the Math Agent — a specialist in arithmetic operations.
    Uses MCP math tools (add, multiply, divide) for precise computation.
    """
    return Agent(
        name="MathAgent",
        description=(
            "Specialist in mathematical computations. Can add, multiply, and "
            "divide numbers with precision using dedicated math tools."
        ),
        instructions=(
            "You are a Math Agent. You perform mathematical operations by using your "
            "available tools: add, multiply, and divide. Always use the tools for "
            "calculations — never compute in your head. Show the tool you used and "
            "the result clearly. If a user asks something outside math, say so."
        ),
        client=client,
        tools=mcp_tool,
    )


def create_string_agent(client: OpenAIChatClient, mcp_tool: MCPStdioTool) -> Agent:
    """
    Create the String Agent — a specialist in string manipulation.
    Uses MCP string tools (to_uppercase, to_lowercase, reverse) for processing.
    """
    return Agent(
        name="StringAgent",
        description=(
            "Specialist in string manipulation. Can convert text to uppercase, "
            "lowercase, or reverse a string using dedicated string tools."
        ),
        instructions=(
            "You are a String Agent. You perform string manipulation operations by "
            "using your available tools: to_uppercase, to_lowercase, and reverse. "
            "Always use the tools for string operations — never transform text manually. "
            "Show the tool you used and the result clearly. If a user asks something "
            "outside string manipulation, say so."
        ),
        client=client,
        tools=mcp_tool,
    )


def create_manager_agent(client: OpenAIChatClient) -> Agent:
    """
    Create the Magentic Manager Agent — orchestrates the workflow.
    Coordinates between Math Agent and String Agent to solve complex tasks.
    """
    return Agent(
        name="MagenticManager",
        description="Orchestrator that coordinates math and string agents to complete tasks.",
        instructions=(
            "You are the Magentic Manager. You coordinate a team of specialist agents:\n"
            "1. MathAgent — handles all mathematical computations (add, multiply, divide)\n"
            "2. StringAgent — handles all string operations (uppercase, lowercase, reverse)\n\n"
            "When given a task:\n"
            "- Break it down into sub-tasks appropriate for each agent\n"
            "- Delegate to the right agent(s)\n"
            "- Synthesize their results into a clear final answer\n"
            "- If a task requires both math and string operations, coordinate both agents\n"
        ),
        client=client,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: A2A AGENT WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════════

# In a production multi-service deployment, each agent would run as an
# independent A2A server. The A2AAgent class wraps a remote endpoint:
#
#   async with A2AAgent(
#       name="MathAgent",
#       url="http://localhost:8001",
#   ) as remote_math_agent:
#       response = await remote_math_agent.run("What is 2+2?")
#
# For this single-process demo, agents are created locally and participate
# directly in the MagenticBuilder workflow. The A2A wrapping below
# demonstrates the pattern for future multi-service deployment.
#
# To expose an agent as an A2A server (using FastAPI or similar):
#
#   from agent_framework.a2a import A2AAgent
#   from a2a.client import A2ACardResolver
#
#   # Server side: expose the agent
#   # (see agent_framework A2A integration docs for full server setup)
#
#   # Client side: connect to a remote agent
#   async with httpx.AsyncClient() as http_client:
#       resolver = A2ACardResolver(httpx_client=http_client, base_url="http://localhost:8001")
#       card = await resolver.get_agent_card()
#       async with A2AAgent(name=card.name, agent_card=card, url="http://localhost:8001") as agent:
#           result = await agent.run("What is 2+2?")


class A2AEventLogger:
    """
    Simulates A2A communication logging for local agents.
    In a real multi-service setup, this would be replaced by actual A2A
    protocol events captured from the network layer.
    """

    @staticmethod
    def log_request(source: str, target: str, query: str) -> None:
        """Log an A2A request event."""
        print_a2a_event(source, target, f"Request: {query[:80]}...")

    @staticmethod
    def log_response(source: str, target: str, response: str) -> None:
        """Log an A2A response event."""
        print_a2a_event(target, source, f"Response: {response[:80]}...")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: MAGENTIC WORKFLOW BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_magentic_workflow(
    math_agent: Agent,
    string_agent: Agent,
    manager_agent: Agent,
):
    """
    Build the Magentic workflow with full observability.

    - intermediate_outputs=True: Enables streaming of all inner agent
      communication, manager plans, and progress ledger updates.
    - max_round_count: Maximum conversation rounds before the workflow ends.
    - max_stall_count: Max stalls (no progress) before forcing advance.
    """
    workflow = MagenticBuilder(
        participants=[math_agent, string_agent],
        intermediate_outputs=True,
        manager_agent=manager_agent,
        max_round_count=10,
        max_stall_count=3,
        max_reset_count=2,
    ).build()

    return workflow


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: AG-UI EVENT STREAM PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

async def process_workflow_events(workflow, task: str) -> list[Message] | None:
    """
    Run the Magentic workflow with streaming and render ALL events to
    the terminal — this is the AG-UI layer for terminal output.

    Events captured:
    ┌────────────────────────────┬──────────────────────────────────────────┐
    │ Event Type                 │ What It Shows                            │
    ├────────────────────────────┼──────────────────────────────────────────┤
    │ executor_invoked           │ Agent starting to process                │
    │ executor_completed         │ Agent finished processing                │
    │ output (AgentResponseUpdate)│ Streaming tokens from an agent          │
    │ magentic_orchestrator      │ Manager plan & progress ledger           │
    │ group_chat (RequestSent)   │ Which agent is being called next         │
    │ output (final)             │ Final conversation transcript            │
    │ superstep_started          │ Workflow superstep beginning             │
    │ superstep_completed        │ Workflow superstep ending                │
    │ data                       │ Intermediate data (tool calls, etc.)     │
    │ error / warning            │ Errors or warnings during execution      │
    └────────────────────────────┴──────────────────────────────────────────┘
    """

    last_response_id: str | None = None
    output_event: WorkflowEvent | None = None
    a2a_logger = A2AEventLogger()

    print_header("WORKFLOW EXECUTION — EVENT STREAM")
    print_event("Task", task, Colors.WHITE)
    print_separator()

    async for event in workflow.run(task, stream=True):

        # ── Streaming Token Output from Agents ───────────────────────────
        if event.type == "output" and isinstance(event.data, AgentResponseUpdate):
            response_id = event.data.response_id
            is_new = response_id != last_response_id

            if is_new and last_response_id is not None:
                print()  # newline after previous agent's output

            print_agent_token(
                executor_id=event.executor_id or "unknown",
                text=str(event.data),
                is_new=is_new,
            )
            last_response_id = response_id

        # ── Magentic Orchestrator Events (Plan / Progress Ledger) ────────
        elif event.type == "magentic_orchestrator":
            if last_response_id is not None:
                print()  # close any open streaming line
                last_response_id = None

            print_separator()
            event_type_name = event.data.event_type.name if hasattr(event.data, 'event_type') else "unknown"
            print_orchestrator_event(event_type_name, "")

            if isinstance(event.data.content, Message):
                plan_text = event.data.content.text or "(empty plan)"
                print(f"    {Colors.BLUE}{plan_text}{Colors.RESET}")

            elif isinstance(event.data.content, MagenticProgressLedger):
                ledger_json = json.dumps(event.data.content.to_dict(), indent=2)
                # Print each line of the ledger indented
                for line in ledger_json.split("\n"):
                    print(f"    {Colors.BLUE}{line}{Colors.RESET}")

            print_separator()

        # ── Group Chat Request Routing ───────────────────────────────────
        elif event.type == "group_chat" and isinstance(event.data, GroupChatRequestSentEvent):
            if last_response_id is not None:
                print()
                last_response_id = None

            participant = event.data.participant_name
            round_idx = event.data.round_index

            print_event(
                f"Round {round_idx}",
                f"Routing request → {Colors.BOLD}{participant}{Colors.RESET}",
                Colors.CYAN,
            )

            # Log A2A-style communication
            a2a_logger.log_request("MagenticManager", participant, task[:60])

        # ── Executor Lifecycle Events ────────────────────────────────────
        elif event.type == "executor_invoked":
            print_event(
                "Executor Invoked",
                f"{event.executor_id}",
                Colors.DIM,
            )

        elif event.type == "executor_completed":
            print_event(
                "Executor Completed",
                f"{event.executor_id}",
                Colors.DIM,
            )

        # ── Superstep Events ─────────────────────────────────────────────
        elif event.type == "superstep_started":
            print_event("Superstep", "Started", Colors.DIM)

        elif event.type == "superstep_completed":
            print_event("Superstep", "Completed", Colors.DIM)

        # ── Data Events (intermediate tool calls, etc.) ──────────────────
        elif event.type == "data":
            if last_response_id is not None:
                print()
                last_response_id = None

            data_str = str(event.data)[:120] if event.data else "(no data)"
            print_event("Data", data_str, Colors.YELLOW)

        # ── Error & Warning Events ───────────────────────────────────────
        elif event.type == "error":
            print_event("ERROR", str(event.data), Colors.RED)

        elif event.type == "warning":
            print_event("WARNING", str(event.data), Colors.YELLOW)

        # ── Final Output (conversation transcript) ───────────────────────
        elif event.type == "output":
            output_event = event

    # Close any open streaming line
    if last_response_id is not None:
        print()

    # Return the final conversation transcript
    if output_event and output_event.data:
        return cast(list[Message], output_event.data)

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: CHAT LAYER (User ←→ MAS Interface)
# ═══════════════════════════════════════════════════════════════════════════════

async def chat_loop() -> None:
    """
    Interactive chat layer between the user and the Magentic workflow.

    This layer:
    1. Accepts user queries in natural language
    2. Routes them through the MagenticBuilder workflow
    3. Streams all events (tool calls, agent communication, etc.) to terminal
    4. Displays the final synthesized answer

    Future enhancement: Add a routing mechanism to send queries directly
    to specialized agents (Math or String) without going through Magentic,
    based on intent classification.
    """

    # ── Print Welcome Banner ─────────────────────────────────────────────
    print(f"""
{Colors.CYAN}{Colors.BOLD}
 ╔══════════════════════════════════════════════════════════════════════════╗
 ║                                                                        ║
 ║   Multi-Agent System (MAS) — Microsoft Agent Framework                 ║
 ║                                                                        ║
 ║   Agents:                                                              ║
 ║     🤖 MagenticManager  — orchestrates the workflow                    ║
 ║     🧮 MathAgent        — add, multiply, divide (MCP tools)            ║
 ║     🔤 StringAgent      — uppercase, lowercase, reverse (MCP tools)    ║
 ║                                                                        ║
 ║   Protocols: MCP (tools) · A2A (communication) · AG-UI (streaming)     ║
 ║                                                                        ║
 ║   Type your query below. Type 'quit' or 'exit' to stop.               ║
 ║                                                                        ║
 ╚══════════════════════════════════════════════════════════════════════════╝
{Colors.RESET}""")

    # ── Create Chat Client ───────────────────────────────────────────────
    print(f"  {Colors.DIM}Initializing OpenAI client ({OPENAI_MODEL} @ {OPENAI_BASE_URL})...{Colors.RESET}")
    client = create_chat_client()

    # ── Create MCP Tool Servers ──────────────────────────────────────────
    print(f"  {Colors.DIM}Starting MCP tool servers...{Colors.RESET}")

    async with (
        create_math_mcp_tool() as math_mcp,
        create_string_mcp_tool() as string_mcp,
    ):
        print(f"  {Colors.GREEN}✓ MathTools MCP server ready (add, multiply, divide){Colors.RESET}")
        print(f"  {Colors.GREEN}✓ StringTools MCP server ready (to_uppercase, to_lowercase, reverse){Colors.RESET}")

        # ── Create Agents ────────────────────────────────────────────────
        math_agent = create_math_agent(client, math_mcp)
        string_agent = create_string_agent(client, string_mcp)
        manager_agent = create_manager_agent(client)

        print(f"  {Colors.GREEN}✓ MathAgent initialized{Colors.RESET}")
        print(f"  {Colors.GREEN}✓ StringAgent initialized{Colors.RESET}")
        print(f"  {Colors.GREEN}✓ MagenticManager initialized{Colors.RESET}")

        # ── Build Workflow ───────────────────────────────────────────────
        print(f"  {Colors.DIM}Building Magentic workflow...{Colors.RESET}")
        workflow = build_magentic_workflow(math_agent, string_agent, manager_agent)
        print(f"  {Colors.GREEN}✓ Workflow ready (intermediate_outputs=True, stream=True){Colors.RESET}")

        print_separator()
        print(f"  {Colors.WHITE}{Colors.BOLD}System ready. Awaiting your queries.{Colors.RESET}\n")

        # ── Interactive Loop ─────────────────────────────────────────────
        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: input(f"  {Colors.CYAN}{Colors.BOLD}You ▶ {Colors.RESET}"),
                )
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {Colors.DIM}Goodbye!{Colors.RESET}")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print(f"\n  {Colors.DIM}Goodbye!{Colors.RESET}")
                break

            # ── Future: Direct Routing (bypass Magentic) ─────────────
            # In future, the chat layer can classify intent and route
            # directly to a specialized agent:
            #
            #   if is_math_query(user_input):
            #       result = await math_agent.run(user_input)
            #   elif is_string_query(user_input):
            #       result = await string_agent.run(user_input)
            #   else:
            #       # Fall through to Magentic
            #

            # ── Route Through Magentic Workflow ──────────────────────
            # Each query gets a fresh workflow to avoid state leakage
            workflow = build_magentic_workflow(math_agent, string_agent, manager_agent)

            final_messages = await process_workflow_events(workflow, user_input)

            # ── Display Final Answer ─────────────────────────────────
            if final_messages:
                print_header("FINAL CONVERSATION TRANSCRIPT")
                for msg in final_messages:
                    author = msg.author_name or msg.role
                    text = msg.text or "(no text)"
                    print(f"  {Colors.BOLD}{author}:{Colors.RESET} {text}\n")
            else:
                print(f"\n  {Colors.YELLOW}No output received from the workflow.{Colors.RESET}")

            print_separator()
            print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        print(f"\n{Colors.DIM}Interrupted. Shutting down...{Colors.RESET}")
