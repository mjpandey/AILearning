from langgraph.graph import StateGraph, END
from graphs.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agents.trading_agent import llm, tools
from agents.planner_agent import planner_agent
from agents.market_analyst import market_analyst
from agents.risk_agent import risk_agent
from agents.decision_agent import decision_agent



builder = StateGraph(AgentState)

builder.add_node("planner", planner_agent)
builder.add_node("market_analyst", market_analyst)
builder.add_node("risk_agent", risk_agent)
builder.add_node("decision_agent", decision_agent)

builder.set_entry_point("planner");

builder.add_edge("planner", "market_analyst")
builder.add_edge("market_analyst", "risk_agent")
builder.add_edge("risk_agent", "decision_agent")
builder.add_edge("decision_agent", END)

app = builder.compile()


