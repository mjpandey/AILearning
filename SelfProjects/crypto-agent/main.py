from graphs.trading_graph import app

query = "Should I long BTC today?"

result = app.invoke({"query": query})

print(result)