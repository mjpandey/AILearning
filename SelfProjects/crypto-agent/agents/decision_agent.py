from agents.trading_agent import llm


def decision_agent(state):

    market = state["market_analysis"]
    risk = state["risk_assessment"]

    prompt = f"""
    Based on the following:

    Market Analysis:
    {market}

    Risk Assessment:
    {risk}

    Provide final trading recommendation.

    Include:
    - Decision
    - Reasoning
    - Confidence
    """

    result = llm.invoke(prompt)

    return {"final_decision": result.content}