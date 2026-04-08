# Multi-Agent System (MAS) — Microsoft Agent Framework

A fully observable multi-agent system built with the [Microsoft Agent Framework (MAF)](https://learn.microsoft.com/en-us/agent-framework/overview/?pivots=programming-language-python), demonstrating MCP tool integration, A2A communication patterns, and AG-UI event streaming.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 Terminal UI (Chat Layer)                  │
│          AG-UI Event Stream → Rich Terminal Output        │
└─────────────────────┬────────────────────────────────────┘
                      │
        ┌─────────────▼──────────────┐
        │    MagenticBuilder         │
        │  (Magentic Manager Agent)  │
        │  intermediate_outputs=True │
        │  stream=True               │
        └──────┬───────────┬─────────┘
               │           │
    ┌──────────▼───┐  ┌───▼──────────┐
    │  Math Agent  │  │ String Agent │
    │  (MCP tools) │  │  (MCP tools) │
    └──────┬───────┘  └───┬──────────┘
           │              │
    ┌──────▼───────┐  ┌───▼──────────┐
    │  MCP Server  │  │  MCP Server  │
    │  (math_srv)  │  │  (string_srv)│
    │  add,mul,div │  │  upper,lower │
    │              │  │  reverse     │
    └──────────────┘  └──────────────┘
```

## Agents

| Agent | Role | Tools (MCP) |
|-------|------|-------------|
| **MagenticManager** | Orchestrates the workflow, plans tasks, delegates to specialists | — |
| **MathAgent** | Arithmetic operations | `add`, `multiply`, `divide` |
| **StringAgent** | String manipulation | `to_uppercase`, `to_lowercase`, `reverse` |

## Protocols Used

| Protocol | Purpose |
|----------|---------|
| **MCP** (Model Context Protocol) | Tool integration via stdio subprocess servers |
| **A2A** (Agent-to-Agent) | Inter-agent communication patterns (local demo, production-ready for remote) |
| **AG-UI** | Real-time event streaming to UI (terminal in this demo) |

## Prerequisites

- Python 3.10+
- OpenAI API key (or any OpenAI-compatible API)

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# or: venv\Scripts\activate  # Windows
```

### 2. Install dependencies

```bash
pip install agent-framework --pre
pip install agent-framework-openai --pre
pip install agent-framework-a2a --pre
pip install agent-framework-ag-ui --pre
pip install agent-framework-orchestrations --pre
pip install "mcp[cli]" --pre
pip install python-dotenv httpx
```

### 3. Configure environment

Edit `.env` and set your OpenAI API key:

```
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-5.1
```

### 4. Run

```bash
python main.py
```

## Usage

Once running, the interactive chat loop accepts natural language queries:

```
You ▶ What is 15 multiplied by 7 plus 23?
You ▶ Reverse the string "hello world" and convert to uppercase
You ▶ Add 10 and 20, then reverse the result as a string
You ▶ quit
```

## What You See in the Terminal

The system streams ALL internal events in real time:

| Event | Description |
|-------|-------------|
| 🤖 Agent Tokens | Streaming text output from each agent as it generates |
| 📋 Orchestrator Plan | Manager's task decomposition and planning ledger |
| 🔧 Tool Calls | MCP tool invocations (add, multiply, reverse, etc.) |
| 🔗 A2A Events | Communication between agents (request/response routing) |
| ▸ Executor Events | Agent lifecycle (invoked, completed) |
| ▸ Superstep Events | Workflow execution phases |

## Key MAF Features Used

### `intermediate_outputs=True`

Passed to `MagenticBuilder()` — this enables streaming of:
- All inner agent-to-agent communication
- The manager's planning and progress ledger
- Which agent is being invoked and when

### `workflow.run(task, stream=True)`

Streams **every** workflow event including:
- `executor_invoked` / `executor_completed` — agent lifecycle
- `output` with `AgentResponseUpdate` — streaming tokens
- `magentic_orchestrator` — manager plan and progress ledger
- `group_chat` with `GroupChatRequestSentEvent` — routing decisions
- `superstep_started` / `superstep_completed` — execution phases

### Custom AG-UI Events

For tool-call-level visibility, the system adds logging around MCP tool invocations. In a production UI, these would be emitted as custom `WorkflowEvent` instances with `type="tool_call"` for the frontend to render.

## File Structure

```
MAF_MAS/
├── main.py            # Complete MAS — agents, workflow, chat layer, event stream
├── math_server.py     # MCP server for math tools (add, multiply, divide)
├── string_server.py   # MCP server for string tools (uppercase, lowercase, reverse)
├── .env               # Environment configuration
└── README.md          # This file
```

## Future Enhancements

- **Direct Agent Routing**: Bypass Magentic for simple queries via intent classification in the chat layer
- **Remote A2A Deployment**: Run each agent as an independent A2A server for distributed architecture
- **Web UI**: Replace terminal output with a CopilotKit-based web frontend using AG-UI protocol
- **Tool Approval**: Add human-in-the-loop approval for sensitive tool calls
