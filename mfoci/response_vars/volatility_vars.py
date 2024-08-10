import numpy as np
from mfoci.response_vars.stock_return_vars import load_stock_returns


def load_volatility_index(
    ticker="^VIX", start_date="2004-01-01", end_date="2024-01-01"
):
    """
    Load VIX index data

    :return: pd.DataFrame
    """
    df = load_stock_returns(ticker, start_date=start_date, end_date=end_date)
    df[ticker] = df[ticker].apply(np.log).diff()
    df.dropna(inplace=True)
    return df
