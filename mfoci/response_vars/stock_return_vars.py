import yfinance as yf


def load_stock_returns(tickers, start_date="2013-01-01", end_date="2024-01-01"):
    """
    Load stock returns for a list of tickers from Yahoo Finance.
    """
    df = yf.download(tickers, start=start_date, end=end_date)
    cols = [col for col in df.columns if col[0] == "Adj Close"]
    stock_prices = df[cols]
    stock_prices.columns = [col[1] for col in stock_prices.columns]
    stock_returns = stock_prices.pct_change().dropna()
    return stock_returns
