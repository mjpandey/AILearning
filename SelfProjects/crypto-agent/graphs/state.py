from typing import TypedDict

class AgentState(TypedDict):
    query: str
    result: str
    analysis: str
    messages: list
    market_analysis: str
    risk_assessment: str
    final_decision: str