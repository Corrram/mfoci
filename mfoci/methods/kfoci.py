import pathlib
import subprocess

import pandas as pd


def kfoci(factors, response_vars, r_path, **kwargs):
    """
    Kernel FOCI - implements slide 31 from Jonathan's presentation

    :param factors: pd.DataFrame
    :param response_vars: pd.DataFrame

    :return: list
    """
    # start r subprocess calling
    path = pathlib.Path(__file__).parent
    factors.to_csv(f"{path}/x.csv", index=False)
    response_vars.to_csv(f"{path}/y.csv", index=False)
    filename = str(path / "kfoci.R")
    cmd_list = [r_path, "--vanilla", filename, path]
    if "kernel" in kwargs:
        kernel = kwargs["kernel"]
    else:
        kernel = "rbfdot"
    cmd_list.append(kernel)
    subprocess.call(cmd_list, shell=True)
    selected_cols_df = pd.read_csv(f"{path}/selected_cols.csv")
    selected_cols = selected_cols_df.iloc[:, 0].tolist()
    print(f"KFOCI {kernel} selected indicators:")
    print(selected_cols)
    return selected_cols
