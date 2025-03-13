import numpy as np
import pandas as pd
from mfoci.response_vars.volatility_vars import load_volatility_index


def fake_load_stock_returns(ticker, start_date, end_date):
    # Create a simple DataFrame with a single column for the ticker.
    # The values are chosen so that the log-differences are easy to compute.
    return pd.DataFrame({ticker: [100, 110, 121]})


def test_load_volatility_index(monkeypatch):
    # Patch the load_stock_returns function in the module's namespace.
    monkeypatch.setattr(
        "mfoci.response_vars.volatility_vars.load_stock_returns",
        fake_load_stock_returns,
    )

    ticker = "^VIX"
    # Call the function under test.
    result_df = load_volatility_index(
        ticker=ticker, start_date="2020-01-01", end_date="2020-12-31"
    )

    # Verify that the result is a DataFrame with 2 rows
    # because the first row becomes NaN after diff and is dropped.
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape[0] == 2
    assert result_df.shape[1] == 1

    # Calculate expected log-differences:
    # For the second row: np.log(110) - np.log(100)
    # For the third row: np.log(121) - np.log(110)
    expected_first = np.log(110) - np.log(100)
    expected_second = np.log(121) - np.log(110)

    # Use numpy.testing to compare floating point values
    np.testing.assert_allclose(result_df[ticker].iloc[0], expected_first, rtol=1e-5)
    np.testing.assert_allclose(result_df[ticker].iloc[1], expected_second, rtol=1e-5)
