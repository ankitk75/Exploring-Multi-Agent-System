# Multi-Agent System — Microsoft Agent Framework

A production-ready **Multi-Agent System (MAS)** built with Microsoft Agent Framework (MAF), demonstrating:

- **MagenticBuilder** orchestrating multiple specialised agents
- **MCP** (Model Context Protocol) for tool exposure
- **A2A** (Agent-to-Agent) protocol for inter-agent communication
- **AG-UI** event streaming for real-time observability (terminal → future UI)
- A **Chat Layer** that routes queries to the right execution path

---

## Architecture

```
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
│  │  MathAgent  (A2A)  │   │        └─────────────────────────────┘
│  │  MCP: add/mul/div  │   │
│  ├────────────────────┤   │
│  │  StringAgent (A2A) │   │
│  │  MCP: up/lo/rev    │   │
│  └────────────────────┘   │
└───────────────────────────┘
```

### Components

| Component | Role | Key tech |
|---|---|---|
| **Chat Layer** | Routes user input to the right execution path | Keyword router (swap with classifier in prod) |
| **MagenticManager** | Orchestrates multi-step tasks across participants | `MagenticBuilder`, `intermediate_outputs=True` |
| **MathAgent** | Arithmetic operations | MAF `Agent`, MCP `add` / `multiply` / `divide` |
| **StringAgent** | String transformations | MAF `Agent`, MCP `to_uppercase` / `to_lowercase` / `reverse_string` |
| **A2A Registry** | Inter-agent message routing | `A2AAgent`, `A2ACardResolver` |
| **AG-UI Renderer** | Real-time event display | `CustomEvent`, `ToolCallStart/End`, `TextMessage*` |

---

## AG-UI Event Streaming — Key Design Decision

> **Q: Is `intermediate_outputs=True` + `stream=True` alone enough to show the full inner workings in a UI?**

**Short answer: No — you need both built-in streaming AND custom AG-UI events.**

| What you get for free | What requires custom events |
|---|---|
| Each participant's final response | A2A manager → participant dispatches |
| MagenticBuilder turn-by-turn loop | Tool-call start / args / result |
| Final workflow answer | Agent reasoning traces |

The code addresses this with:

1. `intermediate_outputs=True` → surfaces per-participant output in the MAF event stream
2. `workflow.run(task, stream=True)` → yields events as they arrive
3. `_adapt_event()` → translates MAF-native events to AG-UI protocol events
4. `on_message` callback on `MagenticBuilder` → fires for every A2A dispatch; we emit `CustomEvent("a2a_message")`
5. `on_tool_call` / `on_tool_result` hooks on each agent → emit `ToolCallStartEvent` / `ToolCallEndEvent`

---

## Requirements

### Python
- Python 3.11+

### Packages

```txt
# Core framework
agent-framework          # Microsoft Agent Framework (MAF)
agent-framework-ag-ui    # MAF ↔ AG-UI integration
ag-ui-protocol           # AG-UI event types

# A2A
a2a-sdk                  # Agent-to-Agent protocol SDK

# MCP
mcp                      # Model Context Protocol SDK (FastMCP included)

# LLM
openai                   # OpenAI-compatible client (used by MAF internally)

# Utilities
python-dotenv
rich
```

Install everything:

```bash
pip install agent-framework agent-framework-ag-ui ag-ui-protocol \
            a2a-sdk mcp openai python-dotenv rich
```

---

## Setup

### 1. Clone / copy the project

```
mas_system.py
.env
```

### 2. Configure your `.env`

```bash
cp .env.example .env
# Edit .env with your API key and model settings
```

Minimum required:

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1   # or your custom endpoint
MODEL_NAME=gpt-4o
```

### 3. Run

```bash
python mas_system.py
```

---

## Usage

### Interactive REPL

```
You > What is 42 multiplied by 7?
You > Reverse the string 'Hello World'
You > Multiply 6 by 9, then reverse the result as a string
You > demo          # runs all built-in demo queries
You > agents        # lists registered A2A agents
You > exit
```

### Routing Logic

| Query pattern | Route | Agents involved |
|---|---|---|
| Purely arithmetic (`add`, `multiply`, `divide`, `plus`, …) | Direct → MathAgent | ChatLayer → MathAgent (A2A) |
| Purely string (`uppercase`, `reverse`, `text`, …) | Direct → StringAgent | ChatLayer → StringAgent (A2A) |
| Combined or ambiguous | Magentic workflow | MagenticManager → Math + String (A2A) |

---

## Terminal Output Guide

| Symbol | Meaning |
|---|---|
| `▶ RUN STARTED` | Magentic workflow begins |
| `💬 [AgentName]` | Agent natural-language reply |
| `🔧 TOOL CALL [Agent] → tool_name` | MCP tool invocation |
| `args: {...}` | Tool input arguments |
| `result: ...` | Tool output |
| `A2A From → To` (cyan panel) | Inter-agent A2A message |
| `⚙ Magentic[step]` | MagenticBuilder intermediate step |
| `■ RUN FINISHED` | Workflow complete |

---

## Extending the System

### Add a new agent

1. Create an MCP server section with `FastMCP` and `@mcp_server.tool()` decorators
2. Write a wrapper class that extends `AGUIToolHooksMixin`
3. Add the agent to `MagenticBuilder(participants=[..., new_agent.agent])`
4. Register it: `A2A_REGISTRY.register("NewAgent", new_agent.a2a_agent, new_agent.card)`
5. Add routing keywords to `ChatLayer._MATH_KEYWORDS` / add a new domain set

### Swap to a real UI

Replace the `emit()` function body with a WebSocket push:

```python
def emit(event: Any) -> None:
    websocket.send(event.model_dump_json())
```

All AG-UI events are already serialisable via Pydantic.

### Distributed deployment

Each agent wrapper is designed to run as an independent HTTP service. Start them with:

```bash
agent-framework serve MathAgentWrapper   --port 8001
agent-framework serve StringAgentWrapper --port 8002
```

Update `.env` with the real URLs and replace `InProcessA2ARegistry` with a network-aware `A2AClient`.

---

## File Structure

```
mas_system.py          ← single-file MAS implementation
.env                   ← your secrets (gitignored)
.env.example           ← template
README.md              ← this file
```

---

## Framework References

- [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/overview/?pivots=programming-language-python)
- [A2A Protocol](https://google.github.io/A2A/)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- [AG-UI Protocol](https://docs.ag-ui.com/)
