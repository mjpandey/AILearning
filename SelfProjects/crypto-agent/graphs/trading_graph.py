from langgraph.graph import StateGraph
from agents.trading_agent import tradingAgent
from graphs.state import AgentState
from utils.prompt_loader import load_prompt


system_prompt = load_prompt()

def analyze_queryAgent(state: AgentState):

    query = state["query"]
    print("System Prompt:\n", system_prompt)
    print("User Query:\n", query)   

    analyze_query = "\n system_prompt:\n"  + system_prompt + "\n\nUser Query:\n" + query

    result = tradingAgent.run(analyze_query)
    return {"result": result}

graph = StateGraph(AgentState)

graph.add_node("analyze", analyze_queryAgent)

graph.set_entry_point("analyze")

app = graph.compile()