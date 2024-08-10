import contextlib

import pandas as pd


def get_engle_sentiment_index(
    columns: str = None,
    shift: int = 1,
    precision: int = 6,
    multiplying_factor: float = 100.0,
) -> pd.DataFrame:
    col_names = ["Date", "wsj", "chneg"]
    df = pd.read_excel("Engle_CCN_index_data.xlsx", usecols="A,C,E", names=col_names)
    df["Date"] = pd.to_datetime(df["Date"] * 100 + 1, format="%Y%m%d") + pd.DateOffset(
        months=shift
    )
    df["wsj"] *= multiplying_factor
    df["chneg"] *= multiplying_factor
    rounded_df = df.round(precision).dropna()
    if columns is not None:
        return rounded_df[["Date"] + [columns]].dropna()
    # interpolate missing days linear
    with contextlib.suppress(ValueError):
        rounded_df.set_index("Date", inplace=True)
        interpolated_df = rounded_df.resample("D").interpolate(method="linear")
    return interpolated_df.dropna()


def ar1_transformation(column):
    # Shift the column values by 1 to get lagged values
    lagged_column = column.shift(1)
    # Subtract lagged values from original values
    transformed_column = column - lagged_column
    # Replace NaN with 0 (for the first row)
    transformed_column.fillna(0, inplace=True)
    return transformed_column


def get_mccc_sentiment_index(
    columns="Overall", precision: int = None, shift: int = 0, ar1: bool = False
) -> pd.DataFrame:
    df = pd.read_csv("mccc_data.csv", parse_dates=["Date"])
    df["Date"] += pd.DateOffset(days=shift)
    if precision is not None:
        df = df.round(precision)
    if isinstance(columns, list):
        df = df[["Date"] + columns]
    elif isinstance(columns, str):
        df = df[["Date"] + [columns]]
    df.set_index("Date", inplace=True)
    df = df.loc[df["Overall"] != 0]
    # perform auto-regression
    if ar1:
        ar1_df = df.copy()
        for column in df.columns:
            ar1_df[column] = ar1_transformation(df[column])
    else:
        ar1_df = df
    ar1_df.rename({"Overall": "mccc"}, axis=1, inplace=True)
    return ar1_df
