"""
Microsoft Agent Framework (MAF) - Multi-Agent System Server
============================================================

A highly observable multi-agent system utilizing:
  - MAF (Microsoft Agent Framework) for agent construction & orchestration
  - MCP (Model Context Protocol) for tool execution via stdio servers
  - A2A (Agent-to-Agent) protocol for inter-agent delegation
  - AG-UI protocol for real-time frontend SSE streaming

Architecture (4 Agents):
  1. Chat Agent      - Entry point; receives user queries, delegates via A2A
  2. Orchestrator    - Manager agent; plans, routes, and synthesizes answers
  3. Math Agent      - Participant; equipped with MCP math tools
  4. String Agent    - Participant; equipped with MCP string tools

LLM Configuration:
  Uses OpenAIChatClient pointing to a custom endpoint (OpenAI GPT-5.1).
  Set OPENAI_API_KEY and optionally OPENAI_BASE_URL in your .env file.

Usage:
  python main.py
  Then POST to http://localhost:8000/api/agent/run with AG-UI RunAgentInput
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Microsoft Agent Framework imports
# ---------------------------------------------------------------------------
from agent_framework import Agent, MCPStdioTool, tool, AgentResponse, AgentResponseUpdate

# MAF OpenAI client - configured for custom endpoint (GPT-5.1)
from agent_framework.openai import OpenAIChatClient

# MAF A2A protocol support - for inter-agent communication
from agent_framework_a2a import A2AAgent

# MAF AG-UI protocol support - for real-time UI streaming
from agent_framework_ag_ui import (
    AgentFrameworkAgent,
    add_agent_framework_fastapi_endpoint,
)

# AG-UI Protocol SDK - event types and encoder for SSE
from ag_ui.core import (
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
    StepStartedEvent,
    StepFinishedEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    CustomEvent,
)
from ag_ui.encoder import EventEncoder

# ---------------------------------------------------------------------------
# Load environment variables from .env file
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Logging configuration for maximum observability
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mas_server")

# ============================================================================
# 1. LLM CONFIGURATION (OpenAIChatClient)
# ============================================================================
# All agents use OpenAIChatClient configured for your custom endpoint.
# The client points to the OpenAI API (or compatible endpoint) with GPT-5.1.
#
# Environment variables:
#   OPENAI_API_KEY  - Your OpenAI API key (required)
#   OPENAI_BASE_URL - Custom base URL (optional, defaults to OpenAI's API)
#   OPENAI_MODEL    - Model name (optional, defaults to gpt-4.1-nano)
# ============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)  # e.g. "https://api.openai.com/v1"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

if not OPENAI_API_KEY:
    logger.warning(
        "OPENAI_API_KEY not set! Create a .env file with your key. "
        "Example: OPENAI_API_KEY=sk-..."
    )


def create_chat_client() -> OpenAIChatClient:
    """Create an OpenAIChatClient configured for GPT-5.1 via custom endpoint.

    This client is shared by all 4 agents in the system. It connects to the
    OpenAI-compatible API specified by environment variables.

    Returns:
        Configured OpenAIChatClient instance.
    """
    kwargs = {
        "model": OPENAI_MODEL,
        "api_key": OPENAI_API_KEY,
    }

    # If a custom base URL is provided, use it (e.g., for proxy or local endpoint)
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL

    client = OpenAIChatClient(**kwargs)
    logger.info(
        f"OpenAIChatClient configured → model={OPENAI_MODEL}, "
        f"base_url={OPENAI_BASE_URL or 'default (api.openai.com)'}"
    )
    return client


# ============================================================================
# 2. MCP TOOL LAYER
# ============================================================================
# Tools are consumed via the Model Context Protocol running locally.
# MCPStdioTool launches each MCP server as a subprocess and communicates
# via stdin/stdout using MCP's JSON-RPC protocol.
#
# Math Tools: add, multiply, divide (from mcp_servers/math_server.py)
# String Tools: convert_to_uppercase, convert_to_lowercase, reverse_string
#               (from mcp_servers/string_server.py)
# ============================================================================

# Resolve the absolute path to the MCP server scripts
MCP_SERVERS_DIR = Path(__file__).parent / "mcp_servers"
PYTHON_EXECUTABLE = sys.executable  # Use the same Python as the current process


def create_math_mcp_tool() -> MCPStdioTool:
    """Create an MCPStdioTool that connects to the Math MCP server.

    The tool launches math_server.py as a subprocess and exposes
    all its tools (add, multiply, divide) to the agent.
    """
    math_server_path = str(MCP_SERVERS_DIR / "math_server.py")
    logger.info(f"Creating MCP Math Tool → {math_server_path}")

    return MCPStdioTool(
        name="math_tools",
        command=PYTHON_EXECUTABLE,
        args=[math_server_path],
        description="Mathematical operations: add, multiply, and divide numbers.",
    )


def create_string_mcp_tool() -> MCPStdioTool:
    """Create an MCPStdioTool that connects to the String MCP server.

    The tool launches string_server.py as a subprocess and exposes
    all its tools (convert_to_uppercase, convert_to_lowercase, reverse_string)
    to the agent.
    """
    string_server_path = str(MCP_SERVERS_DIR / "string_server.py")
    logger.info(f"Creating MCP String Tool → {string_server_path}")

    return MCPStdioTool(
        name="string_tools",
        command=PYTHON_EXECUTABLE,
        args=[string_server_path],
        description="String manipulation operations: uppercase, lowercase, and reverse.",
    )


# ============================================================================
# 3. AGENT TOPOLOGY (4 Agents using MAF)
# ============================================================================
# All agents are instances of agent_framework.Agent. They communicate with
# each other exclusively in natural language.
#
# Agent 1: Math Agent     - Participant, equipped with MCP math tools
# Agent 2: String Agent   - Participant, equipped with MCP string tools
# Agent 3: Orchestrator   - Manager, plans and routes to participants
# Agent 4: Chat Agent     - Entry point, delegates to Orchestrator via A2A
# ============================================================================


def create_math_agent(client: OpenAIChatClient) -> Agent:
    """Create the Math Agent equipped with MCP math tools.

    This agent specializes in numerical calculations. It receives tasks
    in natural language from the Orchestrator and uses its MCP tools
    (add, multiply, divide) to perform calculations.
    """
    return Agent(
        client=client,
        name="math_agent",
        description="Specializes in mathematical calculations using add, multiply, and divide tools.",
        instructions=(
            "You are the Math Agent in a multi-agent system built with the "
            "Microsoft Agent Framework (MAF). Your role is to handle numerical "
            "calculations. You have access to three math tools via MCP:\n"
            "  - add(a, b): Add two numbers\n"
            "  - multiply(a, b): Multiply two numbers\n"
            "  - divide(a, b): Divide a by b\n\n"
            "When you receive a task, use the appropriate tool(s) to compute the "
            "answer. Always show your work and return the final result clearly. "
            "If the task doesn't involve math, say so and suggest the String Agent."
        ),
        tools=[create_math_mcp_tool()],
    )


def create_string_agent(client: OpenAIChatClient) -> Agent:
    """Create the String Agent equipped with MCP string tools.

    This agent specializes in text manipulation. It receives tasks in
    natural language from the Orchestrator and uses its MCP tools
    (convert_to_uppercase, convert_to_lowercase, reverse_string).
    """
    return Agent(
        client=client,
        name="string_agent",
        description="Specializes in text manipulation using uppercase, lowercase, and reverse tools.",
        instructions=(
            "You are the String Agent in a multi-agent system built with the "
            "Microsoft Agent Framework (MAF). Your role is to handle text "
            "manipulation tasks. You have access to three string tools via MCP:\n"
            "  - convert_to_uppercase(text): Convert text to uppercase\n"
            "  - convert_to_lowercase(text): Convert text to lowercase\n"
            "  - reverse_string(text): Reverse the text\n\n"
            "When you receive a task, use the appropriate tool(s) to transform "
            "the text. Return the result clearly. If the task doesn't involve "
            "text manipulation, say so and suggest the Math Agent."
        ),
        tools=[create_string_mcp_tool()],
    )


def create_orchestrator_agent(
    client: OpenAIChatClient,
    math_agent: Agent,
    string_agent: Agent,
) -> Agent:
    """Create the Orchestrator Agent that manages participant agents.

    The Orchestrator acts as the manager of the multi-agent system.
    It analyzes incoming tasks, plans the execution strategy, routes
    sub-tasks to the appropriate participant agents (Math or String),
    and synthesizes the final answer.

    The participant agents are provided as tools (via their .as_tool()
    or by wrapping them) so the Orchestrator can delegate to them.
    """
    # Create tool functions that delegate to participant agents
    @tool
    async def delegate_to_math_agent(task: str) -> str:
        """Delegate a mathematical calculation task to the Math Agent.

        Use this when the user's request involves numbers, arithmetic,
        or mathematical operations (addition, multiplication, division).

        Args:
            task: A natural language description of the math task to perform.

        Returns:
            The Math Agent's response with the calculation result.
        """
        logger.info(f"[Orchestrator] Delegating to Math Agent: {task}")
        response: AgentResponse = await math_agent.run(task)
        result = response.message.content if response.message else str(response)
        logger.info(f"[Orchestrator] Math Agent responded: {result}")
        return result

    @tool
    async def delegate_to_string_agent(task: str) -> str:
        """Delegate a text manipulation task to the String Agent.

        Use this when the user's request involves text transformation
        such as converting to uppercase, lowercase, or reversing a string.

        Args:
            task: A natural language description of the text manipulation task.

        Returns:
            The String Agent's response with the transformed text.
        """
        logger.info(f"[Orchestrator] Delegating to String Agent: {task}")
        response: AgentResponse = await string_agent.run(task)
        result = response.message.content if response.message else str(response)
        logger.info(f"[Orchestrator] String Agent responded: {result}")
        return result

    return Agent(
        client=client,
        name="orchestrator_agent",
        description="Manager agent that plans, routes tasks to specialist agents, and synthesizes results.",
        instructions=(
            "You are the Orchestrator Agent in a multi-agent system built with "
            "the Microsoft Agent Framework (MAF). You are the central manager.\n\n"
            "YOUR ROLE:\n"
            "1. Analyze the user's request to understand what needs to be done\n"
            "2. Plan the execution by breaking complex tasks into sub-tasks\n"
            "3. Route each sub-task to the appropriate specialist agent:\n"
            "   - delegate_to_math_agent: For arithmetic (add, multiply, divide)\n"
            "   - delegate_to_string_agent: For text ops (uppercase, lowercase, reverse)\n"
            "4. Synthesize results from all agents into a clear final answer\n\n"
            "IMPORTANT RULES:\n"
            "- Always delegate to the specialist agents; do NOT try to do math or "
            "  string manipulation yourself\n"
            "- If a task involves both math AND string operations, delegate both "
            "  parts separately and combine the results\n"
            "- Communicate all delegation results naturally back to the user\n"
            "- Always explain what you did and why"
        ),
        tools=[delegate_to_math_agent, delegate_to_string_agent],
    )


def create_chat_agent(
    client: OpenAIChatClient,
    orchestrator: Agent,
) -> Agent:
    """Create the Chat Agent — the user-facing entry point.

    The Chat Agent receives the user's initial query and delegates it
    to the Orchestrator via A2A-style handoff. It then returns the
    Orchestrator's response back to the user.

    A2A PROTOCOL NOTES:
    -------------------
    In a full distributed deployment, the Chat Agent would connect to the
    Orchestrator via HTTP using the A2A protocol (agent_framework_a2a.A2AAgent).
    For this single-process demo, the A2A delegation is implemented as a
    direct function call to the Orchestrator, following the same contract:
    the Chat Agent sends a natural language query and receives a natural
    language response. This simulates the A2A request/response pattern.
    """

    @tool
    async def a2a_delegate_to_orchestrator(query: str) -> str:
        """Hand off the user's query to the Orchestrator Agent via A2A protocol.

        This implements A2A-style delegation: the Chat Agent sends the full
        user query to the Orchestrator, which plans, routes to specialist
        agents, and returns a synthesized answer.

        In a distributed system, this would be an HTTP POST to the
        Orchestrator's A2A endpoint. Here it's an in-process call
        following the same protocol contract.

        Args:
            query: The user's natural language query to process.

        Returns:
            The Orchestrator's complete, synthesized response.
        """
        logger.info(f"[Chat Agent → A2A → Orchestrator] Handing off: {query}")
        response: AgentResponse = await orchestrator.run(query)
        result = response.message.content if response.message else str(response)
        logger.info(f"[Chat Agent ← A2A ← Orchestrator] Received result")
        return result

    return Agent(
        client=client,
        name="chat_agent",
        description="User-facing entry point that delegates to the Orchestrator via A2A.",
        instructions=(
            "You are the Chat Agent — the user-facing entry point of a "
            "multi-agent system built with the Microsoft Agent Framework (MAF).\n\n"
            "YOUR ROLE:\n"
            "1. Receive the user's query\n"
            "2. Delegate it to the Orchestrator Agent using the "
            "   a2a_delegate_to_orchestrator tool (A2A protocol handoff)\n"
            "3. Present the Orchestrator's response to the user in a clear, "
            "   friendly manner\n\n"
            "RULES:\n"
            "- ALWAYS use the a2a_delegate_to_orchestrator tool to process queries\n"
            "- Do NOT try to answer questions yourself\n"
            "- You may add a brief introduction or formatting to the response\n"
            "- If the Orchestrator returns an error, explain it politely"
        ),
        tools=[a2a_delegate_to_orchestrator],
    )


# ============================================================================
# 5. AG-UI STREAMING + SSE ENDPOINT
# ============================================================================
# The AG-UI protocol provides real-time streaming of ALL agent activities
# to the frontend. We configure maximum verbosity to capture:
#   - Lifecycle events (RunStart, RunEnd)
#   - Agent messaging (TextMessageStart, TextMessageContent, TextMessageEnd)
#   - Step executions (StepStarted, StepFinished)
#   - Tool calls (ToolCallStart, ToolCallArgs, ToolCallEnd, ToolCallResult)
#   - Internal delegations via CustomEvent
#
# Two modes are provided:
#   1. Built-in AG-UI endpoint via add_agent_framework_fastapi_endpoint
#   2. Custom verbose SSE endpoint for maximum observability
# ============================================================================


def create_app() -> FastAPI:
    """Create the FastAPI application with AG-UI SSE endpoints.

    Returns:
        Configured FastAPI app with CORS and AG-UI streaming.
    """
    app = FastAPI(
        title="MAF Multi-Agent System with AG-UI",
        description=(
            "A highly observable multi-agent system using "
            "Microsoft Agent Framework, MCP, A2A, and AG-UI."
        ),
        version="1.0.0",
    )

    # Enable CORS for frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Initialize the LLM client and agent constellation
    # -----------------------------------------------------------------------
    client = create_chat_client()

    # Create participant agents with MCP tools
    math_agent = create_math_agent(client)
    string_agent = create_string_agent(client)

    # Create manager agent with delegation tools
    orchestrator = create_orchestrator_agent(client, math_agent, string_agent)

    # Create user-facing agent with A2A handoff
    chat_agent = create_chat_agent(client, orchestrator)

    logger.info("All 4 agents created successfully:")
    logger.info("  1. chat_agent      → Entry point (user-facing)")
    logger.info("  2. orchestrator    → Manager (plans & routes)")
    logger.info("  3. math_agent      → Participant (MCP math tools)")
    logger.info("  4. string_agent    → Participant (MCP string tools)")

    # -----------------------------------------------------------------------
    # AG-UI Built-in Endpoint
    # -----------------------------------------------------------------------
    # MAF provides a built-in AG-UI adapter via add_agent_framework_fastapi_endpoint.
    # This automatically converts agent run events into AG-UI SSE events.
    # -----------------------------------------------------------------------
    ag_ui_agent = AgentFrameworkAgent(
        agent=chat_agent,
        name="mas_chat_agent",
        description="Multi-Agent System entry point with AG-UI streaming",
        require_confirmation=False,  # Don't require UI confirmation for tool calls
    )

    add_agent_framework_fastapi_endpoint(
        app=app,
        agent=ag_ui_agent,
        path="/api/agent/run",
        allow_origins=["*"],
    )
    logger.info("AG-UI endpoint registered at POST /api/agent/run")

    # -----------------------------------------------------------------------
    # Custom Verbose SSE Endpoint
    # -----------------------------------------------------------------------
    # This endpoint provides MAXIMUM verbosity by manually emitting every
    # AG-UI event type, including internal delegations and MCP payloads.
    # -----------------------------------------------------------------------
    @app.post("/api/agent/run/verbose")
    async def verbose_agent_run(request: Request):
        """Custom AG-UI SSE endpoint with maximum verbosity.

        Streams ALL internal events including:
        - RunStarted / RunFinished / RunError
        - StepStarted / StepFinished (per agent)
        - TextMessageStart / TextMessageContent / TextMessageEnd
        - ToolCallStart / ToolCallArgs / ToolCallEnd / ToolCallResult
        - Custom events for internal A2A delegations

        This endpoint accepts a JSON body with a 'query' field:
        {"query": "Add 5 and 3, then convert the result to uppercase"}
        """
        body = await request.json()
        query = body.get("query", "")
        thread_id = body.get("thread_id", str(uuid.uuid4()))
        run_id = str(uuid.uuid4())

        encoder = EventEncoder()

        async def event_stream() -> AsyncGenerator[str, None]:
            """Generate AG-UI events as SSE data."""

            # --- RUN STARTED ---
            yield encoder.encode(
                RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=thread_id,
                    run_id=run_id,
                    timestamp=int(time.time() * 1000),
                )
            )

            # --- STEP: Chat Agent Processing ---
            step_id = "chat_agent_step"
            yield encoder.encode(
                StepStartedEvent(
                    type=EventType.STEP_STARTED,
                    step_name="Chat Agent: Receiving user query",
                    timestamp=int(time.time() * 1000),
                )
            )

            # --- TEXT MESSAGE: Processing notification ---
            msg_id = f"msg_{uuid.uuid4().hex[:8]}"
            yield encoder.encode(
                TextMessageStartEvent(
                    type=EventType.TEXT_MESSAGE_START,
                    message_id=msg_id,
                    role="assistant",
                    timestamp=int(time.time() * 1000),
                )
            )
            yield encoder.encode(
                TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=msg_id,
                    delta=f"Processing your request: \"{query}\"\n\n",
                    timestamp=int(time.time() * 1000),
                )
            )

            # --- TOOL CALL: A2A Delegation ---
            tool_call_id = f"tc_{uuid.uuid4().hex[:8]}"
            yield encoder.encode(
                ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=tool_call_id,
                    tool_call_name="a2a_delegate_to_orchestrator",
                    parent_message_id=msg_id,
                    timestamp=int(time.time() * 1000),
                )
            )
            yield encoder.encode(
                ToolCallArgsEvent(
                    type=EventType.TOOL_CALL_ARGS,
                    tool_call_id=tool_call_id,
                    delta=json.dumps({"query": query}),
                    timestamp=int(time.time() * 1000),
                )
            )

            # --- CUSTOM EVENT: A2A Handoff ---
            yield encoder.encode(
                CustomEvent(
                    type=EventType.CUSTOM,
                    name="a2a_delegation",
                    value={
                        "from": "chat_agent",
                        "to": "orchestrator_agent",
                        "protocol": "A2A",
                        "query": query,
                    },
                    timestamp=int(time.time() * 1000),
                )
            )

            # --- STEP: Orchestrator Planning ---
            yield encoder.encode(
                StepStartedEvent(
                    type=EventType.STEP_STARTED,
                    step_name="Orchestrator: Planning & routing task",
                    timestamp=int(time.time() * 1000),
                )
            )

            # --- Execute the actual agent run ---
            try:
                logger.info(f"[Verbose SSE] Running chat_agent with query: {query}")

                # Run the full agent chain: chat_agent → orchestrator → participants
                response: AgentResponse = await chat_agent.run(query)

                result_text = ""
                if response.message and response.message.content:
                    result_text = response.message.content
                else:
                    result_text = str(response)

                # --- STEP: Orchestrator Complete ---
                yield encoder.encode(
                    StepFinishedEvent(
                        type=EventType.STEP_FINISHED,
                        step_name="Orchestrator: Task complete",
                        timestamp=int(time.time() * 1000),
                    )
                )

                # --- TOOL CALL RESULT ---
                tool_result_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
                yield encoder.encode(
                    ToolCallEndEvent(
                        type=EventType.TOOL_CALL_END,
                        tool_call_id=tool_call_id,
                        timestamp=int(time.time() * 1000),
                    )
                )
                yield encoder.encode(
                    ToolCallResultEvent(
                        type=EventType.TOOL_CALL_RESULT,
                        message_id=tool_result_msg_id,
                        tool_call_id=tool_call_id,
                        content=result_text,
                        role="tool",
                        timestamp=int(time.time() * 1000),
                    )
                )

                # --- TEXT MESSAGE: Final Response ---
                yield encoder.encode(
                    TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        message_id=msg_id,
                        delta=result_text,
                        timestamp=int(time.time() * 1000),
                    )
                )
                yield encoder.encode(
                    TextMessageEndEvent(
                        type=EventType.TEXT_MESSAGE_END,
                        message_id=msg_id,
                        timestamp=int(time.time() * 1000),
                    )
                )

                # --- STEP: Chat Agent Complete ---
                yield encoder.encode(
                    StepFinishedEvent(
                        type=EventType.STEP_FINISHED,
                        step_name="Chat Agent: Response delivered",
                        timestamp=int(time.time() * 1000),
                    )
                )

                # --- RUN FINISHED ---
                yield encoder.encode(
                    RunFinishedEvent(
                        type=EventType.RUN_FINISHED,
                        thread_id=thread_id,
                        run_id=run_id,
                        result=result_text,
                        timestamp=int(time.time() * 1000),
                    )
                )

            except Exception as e:
                logger.error(f"[Verbose SSE] Error: {e}", exc_info=True)
                yield encoder.encode(
                    RunErrorEvent(
                        type=EventType.RUN_ERROR,
                        message=str(e),
                        code="AGENT_ERROR",
                        timestamp=int(time.time() * 1000),
                    )
                )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # -----------------------------------------------------------------------
    # Health check endpoint
    # -----------------------------------------------------------------------
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "framework": "Microsoft Agent Framework (MAF)",
            "agents": ["chat_agent", "orchestrator_agent", "math_agent", "string_agent"],
            "protocols": ["MCP", "A2A", "AG-UI"],
            "model": OPENAI_MODEL,
        }

    return app


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("  Microsoft Agent Framework - Multi-Agent System")
    logger.info("  Protocols: MCP + A2A + AG-UI")
    logger.info("=" * 70)

    app = create_app()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
