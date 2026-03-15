from graphs.trading_multiagent_graph import app
from langchain_core.messages import HumanMessage


query = "Should I long BTC today?"

result= app.invoke({
    "messages":[HumanMessage(content=query)]}) 

print("\n===== FINAL DECISION =====\n")
print(result["final_decision"])
