from agents.trading_agent import llm
from tools.portfolio_tool import get_portfolio_exposure


def risk_agent(state):

    exposure = get_portfolio_exposure(state)

    prompt = f"""
    Evaluate portfolio risk.

    Current exposure:
    {exposure}

    Assess whether increasing BTC position is risky.
    """

    result = llm.invoke(prompt)

    return {"risk_assessment": result.content}