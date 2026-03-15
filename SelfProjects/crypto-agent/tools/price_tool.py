import yfinance as yf

def get_btc_price():
    btc = yf.Ticker("BTC-USD")
    data = btc.history(period="1d")
    price = data["Close"].iloc[-1]

    return f"BTC current price is ${price}"

# print(get_btc_price())