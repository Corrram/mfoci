import pandas as pd


def load_index_returns():
    filename = "MSCI_countries_USD_data.xlsx"
    df = pd.read_excel(filename, usecols="C,D,E")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    countries = ["AR", "BR"]  # , "CH", "ES", "GB", "IT", "SE", "ZA"]
    # ten_countries = df["ISO"].unique()[:10]
    df = df[df["ISO"].isin(countries)]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d %H:%M:%S")
    df.set_index("Date", inplace=True)
    pivot_df = df.pivot(columns="ISO", values="Price Close")
    returns_df = pivot_df.pct_change().dropna()
    # compute correlation matrix of the columns
    corr_matrix = returns_df.corr()
    print(corr_matrix)
    return returns_df
