import warnings
import numpy as np
import pandas as pd
import io
import requests
from zipfile import ZipFile


def get_fama_french_data(start_date="1963-1-1", end_date="2024-01-01"):
    """
    Get Fama French data directly from Kenneth French's website

    :param start_date: Start date in 'YYYY-MM-DD' format
    :param end_date: End date in 'YYYY-MM-DD' format
    :return: pd.DataFrame
    """
    # URL for the 5 Factors data
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Download the zip file
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download data from {url}")

        # Extract and read the CSV file
        try:
            # Read zip file
            zip_file = ZipFile(io.BytesIO(response.content))

            # Get the CSV file name (typically there's only one file)
            file_name = [name for name in zip_file.namelist() if name.endswith('.CSV')][0]

            # Read the file content
            with zip_file.open(file_name) as file:
                content = file.read().decode('utf-8')

            # Skip the header section and find where the data actually starts
            lines = content.split('\n')
            header_row = None
            for i, line in enumerate(lines):
                if 'Mkt-RF' in line:
                    header_row = i
                    break

            if header_row is None:
                raise ValueError("Could not find header row in the CSV file")

            # Read the CSV data, starting from the header row
            ff5 = pd.read_csv(
                io.StringIO('\n'.join(lines[header_row:])),
                index_col=0,
                na_values=["-99.99", "-999"],
            )

        except Exception as e:
            raise ValueError(f"Failed to parse data: {e}")

    # Clean the data
    ff5.index = pd.to_datetime(ff5.index, format="%Y%m%d", errors='coerce')
    ff5 = ff5.loc[~ff5.index.isna()]  # Remove rows with invalid dates

    # Filter by date range
    ff5 = ff5.loc[start_date:end_date]

    # Remove rows that contain "Annual" in the index
    ff5 = ff5[~ff5.index.astype(str).str.contains("Annual")]

    # Generate random data columns
    np.random.seed(1)
    ff5["PLA-Unif"] = np.random.rand(len(ff5))
    ff5["PLA-Gauss"] = np.random.randn(len(ff5))
    ff5["PLA-Exp"] = np.exp(np.random.randn(len(ff5)))

    # Drop RF column
    if "RF" in ff5.columns:
        del ff5["RF"]

    return ff5


if __name__ == '__main__':
    ff5 = get_fama_french_data("1983-7-1", "1993-7-5")
    exit()