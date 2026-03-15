import operator
import time
from typing import Annotated, Sequence, TypedDict, Literal

from langchain.agents import Tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langsmith import traceable

from tools.price_tool import get_btc_price
from tools.news_tool import get_crypto_news
from tools.sentiment_tool import get_market_sentiment
from tools.portfolio_tool import get_portfolio_exposure

# 1. Define State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    metrics: dict  # Custom state purely to hold our performance logs

# 2. Setup LLM
llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="qwen3.5:397b-cloud", # Assuming `qwen2.5:7b` locally
    temperature=0,
    tags=["local_model_qwen"]
)

# 3. Define Tools by Role (Adding tags for LangSmith tool categorization)
analyst_tools = [
    Tool(
        name="BTC_Price",
        func=get_btc_price,
        description="Get current Bitcoin price",
        tags=["market_data"]
    ),
    Tool(
        name="Crypto_News",
        func=get_crypto_news,
        description="Get latest crypto news",
        tags=["market_data"]
    ),
    Tool(
        name="Market_Sentiment",
        func=get_market_sentiment,
        description="Get crypto market sentiment",
        tags=["market_data", "sentiment"]
    )
]

# Risk Agent: Analyzes portfolio safety
risk_tools = [
     Tool(
         name="Portfolio_Exposure",
         func=get_portfolio_exposure,
         description="Check portfolio exposure",
         tags=["risk_data"]
     )
]

# 4. Create Worker Agents
# Analyst Agent: Gathers market data. Using prebuilt react agent for the Tool->Observation loop.
analyst_agent = create_react_agent(
    llm, 
    tools=analyst_tools, 
    state_modifier="You are a senior Market Analyst. Use available tools to fetch market data and answer questions accurately. Only use tools when necessary."
)

# Risk Agent: Analyzes portfolio safety.
risk_agent = create_react_agent(
    llm, 
    tools=risk_tools, 
    state_modifier="You are a strict Risk Manager. Use the portfolio exposure tool to evaluate whether a trade is safe for the user."
)

# Helper node wrappers that invoke the agent and format the return to add to AgentState
@traceable(name="Analyst Node")
def analyst_node(state: AgentState):
    start_time = time.time()
    result = analyst_agent.invoke(state)
    duration = time.time() - start_time
    
    # Initialize metrics if it doesn't exist
    metrics = state.get("metrics", {})
    metrics["analyst_total_calls"] = metrics.get("analyst_total_calls", 0) + 1
    metrics["analyst_total_time"] = metrics.get("analyst_total_time", 0.0) + duration
    
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="Analyst")
        ],
        "metrics": metrics
    }

@traceable(name="Risk Node")
def risk_node(state: AgentState):
    start_time = time.time()
    result = risk_agent.invoke(state)
    duration = time.time() - start_time
    
    # Initialize metrics if it doesn't exist
    metrics = state.get("metrics", {})
    metrics["risk_total_calls"] = metrics.get("risk_total_calls", 0) + 1
    metrics["risk_total_time"] = metrics.get("risk_total_time", 0.0) + duration
    
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="RiskManager")
        ],
        "metrics": metrics
    }

# 5. Create Planner/Supervisor Node
# The planner decides who acts next or if the task is finished.
members = ["Analyst", "RiskManager"]
system_prompt = (
    "You are a supervisor managing a conversation between the following workers: {members}. "
    "Given the following user request, respond with the worker to act next. "
    "Each worker will perform a task and respond with their results and status. "
    "When finished, respond with FINISH."
)

options = ["FINISH"] + members
# We use pydantic for structured routing, Langchain will coerce the output
class Router(TypedDict):
    """Worker to route to next. If no workers are needed, route to FINISH."""
    next: Literal["FINISH", "Analyst", "RiskManager"]

from langchain_core.output_parsers import StrOutputParser

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}. IMPORTANT: Respond with only the EXACT word of your choice and nothing else.")
]).partial(options=str(options), members=", ".join(members))

def parse_routing(text: str) -> dict:
    # Safely extract the intent from the raw string output
    text_upper = text.upper()
    if "ANALYST" in text_upper:
        return {"next": "Analyst"}
    elif "RISKMANAGER" in text_upper:
        return {"next": "RiskManager"}
    else:
        return {"next": "FINISH"}

planner_chain = planner_prompt | llm | StrOutputParser() | parse_routing

def supervisor_node(state: AgentState):
    routing_decision = planner_chain.invoke(state)
    return {"next": routing_decision["next"]}

# 6. Build Graph Architectures
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("Analyst", analyst_node)
workflow.add_node("RiskManager", risk_node)
workflow.add_node("Supervisor", supervisor_node)

# Add edges Let the Supervisor route the workflow
workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["next"],
    {
        "Analyst": "Analyst",
        "RiskManager": "RiskManager",
        "FINISH": END
    }
)
# After completing their work, route back to the Supervisor to determine what is next
workflow.add_edge("Analyst", "Supervisor")
workflow.add_edge("RiskManager", "Supervisor")

# Set entry point
workflow.set_entry_point("Supervisor")

# Compile graph
app = workflow.compile()

# 7. Interactive Loop
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 Multi-Agent System Initialized")
    print("   Architecture: Supervisor ➡️ Analyst / RiskManager")
    print("   Observability: ON (Tracking Latency & Runs via LangSmith)")
    print("="*50)
    
    while True:
        query = input("\nAsk something (or type 'exit'): ")
        if query.lower() == "exit":
            break
            
        print("\n--- Processing ---")
        
        # We start the payload with empty metrics dictionary so the nodes can append to it
        initial_payload = {
            "messages": [HumanMessage(content=query)],
            "metrics": {}
        }
        
        final_metrics = {}
        
        # Stream the graph execution to show the step-by-step thinking
        for s in app.stream(initial_payload):
            # s is a dict with the name of the node that just executed
            if "__end__" not in s:
                node_name = list(s.keys())[0]
                print(f"\n[Node Execution: {node_name}]")
                
                state_update = s[node_name]
                
                # Print the assistant's message if they replied
                if "messages" in state_update:
                    print(state_update["messages"][-1].content)
                
                # Print the routing decision if the supervisor acted
                if "next" in state_update:
                    print(f"Routing task to -> {state_update['next']}")
                    
                # Save the latest tracking metrics for later
                if "metrics" in state_update:
                    final_metrics = state_update["metrics"]
        
        # Print final observability summary for this execution
        print("\n" + "-"*40)
        print("📊 Execution Metrics Summary:")
        if final_metrics:
            for key, val in final_metrics.items():
                if "time" in key:
                    print(f"   - {key.replace('_', ' ').title()}: {val:.2f} seconds")
                else:
                    print(f"   - {key.replace('_', ' ').title()}: {val} calls")
        else:
            print("   - No worker nodes engaged.")
        print("-" * 40 + "\n--- Done ---")
