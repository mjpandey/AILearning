from langgraph.graph import StateGraph
from agents.trading_agent import agent
from graphs.state import AgentState


def analyze_query(state: AgentState):
    result = agent.run(state["query"])
    return {"result": result}

graph = StateGraph(AgentState)

graph.add_node("analyze", analyze_query)

graph.set_entry_point("analyze")

app = graph.compile()