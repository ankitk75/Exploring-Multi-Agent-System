import asyncio
import json
import os
from typing import Any, AsyncGenerator

# Microsoft Agent Framework Imports
from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient
from agent_framework.orchestrations import MagenticBuilder

# AG-UI Integration Imports
# AgentFrameworkAgent wraps any MAF workflow/agent to yield standard AG-UI protocol events
from agent_framework.ag_ui import AgentFrameworkAgent

# ==========================================
# 1. Configuration & Client Setup
# ==========================================

# Replace with your actual OpenAI API credentials and custom endpoint
API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
ENDPOINT = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
MODEL = "gpt-4o"

client = OpenAIChatCompletionClient(
    api_key=API_KEY,
    base_url=ENDPOINT, 
    model=MODEL
)

# ==========================================
# 2. Define Tools for Specialized Agents
# ==========================================

def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divides first number by second number."""
    if b == 0:
        return "Error: Division by zero."
    return a / b

def to_uppercase(s: str) -> str:
    """Converts a string to uppercase."""
    return s.upper()

def to_lowercase(s: str) -> str:
    """Converts a string to lowercase."""
    return s.lower()

def reverse_string(s: str) -> str:
    """Reverses the given string."""
    return s[::-1]

# ==========================================
# 3. Create Specialized Agents
# ==========================================

math_agent = Agent(
    name="MathAgent",
    instructions="You are a math specialist. Use your tools to solve math queries precisely. Only answer math-related parts.",
    client=client,
    tools=[add, multiply, divide]
)

string_agent = Agent(
    name="StringAgent",
    instructions="You are a string manipulation specialist. Use your tools to modify strings exactly as requested.",
    client=client,
    tools=[to_uppercase, to_lowercase, reverse_string]
)

manager_agent = Agent(
    name="MagenticManager",
    instructions="You are the Magentic Manager. Coordinate the MathAgent and StringAgent to solve complex, multi-step problems.",
    client=client
)

# ==========================================
# 4. Build Magentic Orchestration Workflow
# ==========================================

# intermediate_outputs=True is critical: it tells the manager to yield its 
# meta-reasoning and MagenticProgressLedger events into the stream.
magentic_workflow = MagenticBuilder(
    participants=[math_agent, string_agent],
    manager_agent=manager_agent,
    intermediate_outputs=True, 
    max_round_count=10
).build()

# ==========================================
# 5. Define the Top-Level Router Workflow
# ==========================================

class MasterRouterWorkflow:
    """
    Acts as the entry point, evaluating the user's prompt and routing 
    the request to the appropriate agent or orchestration.
    Implements the standard `run_stream` duck-typing protocol required by MAF.
    """
    name = "MasterRouter"
    description = "Routes queries to specialized agents or the Magentic workflow."
    
    def __init__(self, math, string, magentic):
        self.math_agent = math
        self.string_agent = string
        self.magentic_workflow = magentic

    async def run_stream(self, messages: list[Any], *args, **kwargs) -> AsyncGenerator[Any, None]:
        # Extract text from the latest message
        last_msg = messages[-1]
        content = ""
        
        # Handle different message format types (dict or object) depending on frontend payload
        if hasattr(last_msg, "content"):
            content = last_msg.content
        elif isinstance(last_msg, dict):
            content = last_msg.get("content", "")
            
        content_lower = content.lower()
        
        # ----------------------------------
        # Dynamic Routing Logic
        # ----------------------------------
        if content_lower.startswith("math:"):
            print("\n[Router] Routing to Single Agent: MathAgent")
            async for event in self.math_agent.run_stream(messages, *args, **kwargs):
                yield event
                
        elif content_lower.startswith("string:"):
            print("\n[Router] Routing to Single Agent: StringAgent")
            async for event in self.string_agent.run_stream(messages, *args, **kwargs):
                yield event
                
        else: # Defaults to "magentic:" or unrecognized prefixes
            print("\n[Router] Routing to Orchestrator: Magentic Workflow")
            async for event in self.magentic_workflow.run_stream(messages, *args, **kwargs):
                yield event

# Instantiate the Router
router_workflow = MasterRouterWorkflow(math_agent, string_agent, magentic_workflow)

# ==========================================
# 6. Bind to AG-UI and Execute
# ==========================================

async def main():
    # 1. Wrap the Router with AG-UI integration. 
    # This automatically translates native MAF WorkflowEvents into strict AG-UI protocol events.
    ag_ui_bridge = AgentFrameworkAgent(agent=router_workflow)

    # 2. Define our test queries based on the routing rules
    test_queries = [
        "math: What is 56 multiplied by 14?",
        "string: Please reverse the word 'Orchestration'",
        "magentic: First calculate 25 multiplied by 4, then take the result and spell it completely backwards."
    ]

    all_agui_events = []

    for idx, query in enumerate(test_queries):
        print(f"\n--- Executing Query {idx+1} ---")
        print(f"Prompt: {query}")
        
        run_events = []
        # Simulate the payload shape that an AG-UI frontend sends via HTTP
        input_data = {"messages": [{"role": "user", "content": query}]}
        
        # 3. Execute the run and intercept the AG-UI formatted events
        try:
            # run_agent() triggers the router, which triggers the target workflow,
            # which bubbles up MAF events, which are converted to AG-UI events.
            async for ag_ui_event in ag_ui_bridge.run_agent(input_data):
                
                # ag_ui_event is an AG-UI BaseEvent pydantic model. 
                # .model_dump() serializes it to a clean dictionary.
                event_dict = ag_ui_event.model_dump()
                run_events.append(event_dict)
                
                event_type = event_dict.get("type", "UNKNOWN")
                print(f"  -> Emitted AG-UI Event: {event_type}")
                
        except Exception as e:
            print(f"Error during execution: {e}")

        # Store events for this specific run
        all_agui_events.append({
            "query": query,
            "events": run_events
        })

    # ==========================================
    # 7. Save Events to JSON File
    # ==========================================
    output_filename = "ag_ui_events_log.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_agui_events, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Execution complete. All AG-UI protocol events successfully saved to '{output_filename}'")

if __name__ == "__main__":
    asyncio.run(main())