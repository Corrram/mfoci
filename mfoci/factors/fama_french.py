import warnings

import numpy as np
import pandas_datareader as pdr


def get_fama_french_data(start_date="1963-1-1", end_date="2024-01-01"):
    """
    Get Fama French data from Kenneth French's website

    :return: pd.DataFrame
    """
    dataset_name = "F-F_Research_Data_5_Factors_2x3_daily"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ff5_data = pdr.get_data_famafrench(dataset_name, start=start_date, end=end_date)
    ff5 = ff5_data[0].reset_index().dropna()
    # set seed
    np.random.seed(1)
    ff5["PLA-Unif"] = np.random.rand(len(ff5))
    ff5["PLA-Gauss"] = np.random.randn(len(ff5))
    ff5["PLA-Exp"] = np.exp(np.random.randn(len(ff5)))
    # ff5["Date"] = ff5["Date"].dt.strftime("%Y-%m-%d")
    ff5.set_index("Date", inplace=True)
    del ff5["RF"]
    return ff5
