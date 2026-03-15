from langchain.agents import initialize_agent
from langchain.agents import Tool
from utils.prompt_loader import load_prompt
from langchain_openai import ChatOpenAI

from tools.price_tool import get_btc_price
from tools.news_tool import get_crypto_news
from tools.sentiment_tool import get_market_sentiment
from tools.portfolio_tool import get_portfolio_exposure

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="gemma-3-12b-it",
    temperature=0
)

tools = [
    Tool(
        name="BTC Price",
        func=get_btc_price,
        description="Get current Bitcoin price"
    ),
    Tool(
        name="Crypto News",
        func=get_crypto_news,
        description="Get latest crypto news"
    ),
    Tool(
        name="Market Sentiment",
        func=get_market_sentiment,
        description="Get crypto market sentiment"
    ),
    Tool(
        name="Portfolio Exposure",
        func=get_portfolio_exposure,
        description="Check portfolio exposure"
    )
]

system_prompt = load_prompt()

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

# if __name__ == "__main__":
#     question = "What is the current BTC price?"
#     response = agent.run(question)
    # print("\nFINAL ANSWER:\n", response)

if __name__ == "__main__":
    while True:
        query = input("\nAsk something: ")
        if query == "exit":
            break
        response = agent.invoke(query)
        print("\nAnswer:", response)