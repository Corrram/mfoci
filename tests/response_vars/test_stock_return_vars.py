import numpy as np
import pandas as pd
import pytest

from mfoci.response_vars.stock_return_vars import load_stock_returns


def fake_download(tickers, start, end):
    """
    Fake yf.download implementation.

    Returns a DataFrame with MultiIndex columns where the first level is 'Adj Close'
    and the second level is the ticker symbol. We simulate 3 days of data with prices:
    100, 110, 121.
    """
    # Ensure tickers is a list for consistency
    if isinstance(tickers, str):
        tickers = [tickers]
    # Create MultiIndex columns: all columns are ('Adj Close', ticker)
    columns = pd.MultiIndex.from_tuples([("Adj Close", t) for t in tickers])
    # Create a date range (3 dates starting at the given start date)
    dates = pd.date_range(start=start, periods=3)
    # Create data: each ticker gets the same price series [100, 110, 121]
    prices = np.tile(np.array([100, 110, 121]).reshape(3, 1), (1, len(tickers)))
    return pd.DataFrame(prices, index=dates, columns=columns)


@pytest.fixture(autouse=True)
def patch_yf_download(monkeypatch):
    """
    Automatically patch the yfinance.download function in the stock_return_vars module
    so that both tests (for stock returns and volatility) use the fake data.
    """
    monkeypatch.setattr(
        "mfoci.response_vars.stock_return_vars.yf.download", fake_download
    )


def test_load_stock_returns():
    """
    Test the load_stock_returns function.

    The fake download returns 3 rows of prices. The function computes percentage change
    and drops the first row (which is NaN). For prices 100, 110, 121 the returns are:
      row1: (110/100 - 1) = 0.1
      row2: (121/110 - 1) = 0.1
    """
    ticker = "TEST"
    df_returns = load_stock_returns(
        ticker, start_date="2021-01-01", end_date="2021-01-03"
    )

    # Expected index: the first date is dropped because of pct_change -> two dates remain.
    expected_index = pd.date_range(start="2021-01-01", periods=3)[1:]
    expected_df = pd.DataFrame({ticker: [0.1, 0.1]}, index=expected_index)

    pd.testing.assert_frame_equal(df_returns, expected_df)
