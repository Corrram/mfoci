import pandas as pd

from mfoci import filter_for_common_indices


def test_filterer():
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    df2 = pd.DataFrame({"B": [4, 5, 6], "C": [7, 8, 9]}, index=["b", "c", "d"])
    result_df1, result_df2 = filter_for_common_indices(df1, df2)
    assert result_df1.index.tolist() == ["b", "c"]
    assert result_df1.values.tolist() == [[2, 5], [3, 6]]
    assert result_df2.index.tolist() == ["b", "c"]
    assert result_df2.values.tolist() == [[4, 7], [5, 8]]
