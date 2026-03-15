from agents.trading_agent import llm


def planner_agent(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    messages.append(response)
    return {"messages": messages}