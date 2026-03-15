from agents.trading_agent import llm
from tools.price_tool import get_btc_price
from tools.news_tool import get_crypto_news
from tools.sentiment_tool import get_market_sentiment


def market_analyst(state):

    query = state["messages"][-1].content

    price = get_btc_price(state)
    news = get_crypto_news(state)
    sentiment = get_market_sentiment(state)

    prompt = f"""
    Analyze crypto market conditions.

    BTC Price:
    {price}

    News:
    {news}

    Sentiment:
    {sentiment}

    User question:
    {query}

    Provide market analysis.
    """

    result = llm.invoke(prompt)

    return {"market_analysis": result.content}