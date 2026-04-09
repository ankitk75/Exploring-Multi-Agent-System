import asyncio
import json
import os
from typing import Any, AsyncGenerator

# FastAPI Imports for testing
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Microsoft Agent Framework Imports
from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient
from agent_framework.orchestrations import MagenticBuilder

# AG-UI Integration Import (The official public endpoint binder)
from agent_framework.ag_ui import add_agent_framework_fastapi_endpoint

# ==========================================
# 1. Configuration & Client Setup
# ==========================================

# Make sure to set these environment variables before running, 
# or replace the defaults with your actual keys/endpoints.
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
    instructions="You are a math specialist. Use your tools to solve math queries precisely.",
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
    instructions="You are the Magentic Manager. Coordinate the MathAgent and StringAgent to solve complex problems.",
    client=client
)

# ==========================================
# 4. Build Magentic Orchestration Workflow
# ==========================================
# intermediate_outputs=True is critical for exposing MagenticProgressLedger events
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
    Acts as the single entry point. Evaluates the prompt and routes 
    the request to the appropriate single agent or the Magentic orchestrator.
    """
    name = "MasterRouter"
    description = "Routes queries to specialized agents or Magentic workflow."
    
    def __init__(self, math, string, magentic):
        self.math_agent = math
        self.string_agent = string
        self.magentic_workflow = magentic

    async def run_stream(self, messages: list[Any], *args, **kwargs) -> AsyncGenerator[Any, None]:
        # Extract text from the latest message safely
        last_msg = messages[-1]
        content = ""
        
        if hasattr(last_msg, "content"):
            content = last_msg.content
        elif isinstance(last_msg, dict):
            content = last_msg.get("content", "")
            
        content_lower = content.lower()
        
        # --- Routing Logic ---
        if content_lower.startswith("math:"):
            print("\n[Router] 🧭 Routing to Single Agent: MathAgent")
            async for event in self.math_agent.run_stream(messages, *args, **kwargs):
                yield event
                
        elif content_lower.startswith("string:"):
            print("\n[Router] 🧭 Routing to Single Agent: StringAgent")
            async for event in self.string_agent.run_stream(messages, *args, **kwargs):
                yield event
                
        elif content_lower.startswith("magentic:"):
            print("\n[Router] 🧭 Routing to Orchestrator: Magentic Workflow")
            async for event in self.magentic_workflow.run_stream(messages, *args, **kwargs):
                yield event
                
        else:
            # Default fallback if no prefix is matched
            print("\n[Router] 🧭 Defaulting to Orchestrator: Magentic Workflow")
            async for event in self.magentic_workflow.run_stream(messages, *args, **kwargs):
                yield event

router_workflow = MasterRouterWorkflow(math_agent, string_agent, magentic_workflow)

# ==========================================
# 6. Bind to FastAPI and Emulate AG-UI Client
# ==========================================

def main():
    # 1. Create a FastAPI app and bind our Router using the official MAF AG-UI endpoint
    app = FastAPI()
    add_agent_framework_fastapi_endpoint(app, path="/chat", workflow=router_workflow)

    # 2. Use FastAPI's TestClient to simulate real HTTP frontend requests
    test_client = TestClient(app)

    test_queries = [
        "math: What is 56 multiplied by 14?",
        "string: Please reverse the word 'Orchestration'",
        "magentic: Calculate 25 multiplied by 4, then spell the result backwards."
    ]

    all_agui_events = []

    for idx, query in enumerate(test_queries):
        print(f"\n--- Executing Query {idx+1} ---")
        print(f"Prompt: {query}")
        
        run_events = []
        payload = {"messages": [{"role": "user", "content": query}]}
        
        try:
            # 3. Send POST request to our mock AG-UI endpoint
            # This triggers the router and streams back Server-Sent Events (SSE)
            response = test_client.post("/chat", json=payload)
            
            # 4. Parse the SSE stream line by line
            # SSE streams format data as: data: {"type": "...", ...}\n\n
            for line in response.text.splitlines():
                if line.startswith("data: "):
                    event_str = line[6:].strip() # Remove 'data: ' prefix
                    
                    if event_str == "[DONE]":
                        continue
                        
                    # Parse the standard AG-UI JSON event
                    event_json = json.loads(event_str)
                    run_events.append(event_json)
                    
                    event_type = event_json.get("type", "UNKNOWN")
                    print(f"  -> Captured AG-UI Event: {event_type}")
                    
        except Exception as e:
            print(f"Error during execution: {e}")

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
    # TestClient is synchronous, so we don't need asyncio.run() for the main block
    main()
