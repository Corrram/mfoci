import numpy as np
import pandas as pd
from mfoci.response_vars.etf_vars import load_index_returns


def fake_read_excel(filename, usecols):
    data = {
        "Date": [
            "2021-01-01 00:00:00",
            "2021-01-01 00:00:00",
            "2021-01-02 00:00:00",
            "2021-01-02 00:00:00",
            "2021-01-03 00:00:00",
            "2021-01-03 00:00:00",
        ],
        "ISO": ["AR", "BR", "AR", "BR", "AR", "BR"],
        "Price Close": [100, 200, 110, 220, 121, 242],
    }
    return pd.DataFrame(data)


def test_load_index_returns(monkeypatch, capsys):
    # Replace pd.read_excel with our fake function.
    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    # Call the function under test.
    returns_df = load_index_returns()

    # Expected behavior:
    # The fake data creates a pivot table with dates as index:
    # 2021-01-01: AR=100, BR=200
    # 2021-01-02: AR=110, BR=220
    # 2021-01-03: AR=121, BR=242
    # pct_change() drops the first row so the returned DataFrame should have:
    # For 2021-01-02: AR: (110/100 - 1) = 0.1, BR: (220/200 - 1) = 0.1
    # For 2021-01-03: AR: (121/110 - 1) = 0.1, BR: (242/220 - 1) = 0.1
    expected_index = pd.to_datetime(["2021-01-02 00:00:00", "2021-01-03 00:00:00"])
    expected_df = pd.DataFrame(
        {"AR": [0.1, 0.1], "BR": [0.1, 0.1]}, index=expected_index
    )
    expected_df.index.name = (
        "Date"  # Ensure the index name matches the one in the function
    )

    assert np.isclose(returns_df, expected_df).all(), "Unexpected returns DataFrame"

    # Verify that the function printed the correlation matrix.
    captured = capsys.readouterr().out
    assert "NaN" in captured
