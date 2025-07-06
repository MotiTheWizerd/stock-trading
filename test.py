import yfinance as yf

data = yf.download("AAPL", start="2023-01-01", end="2023-12-31")
print(data.head())