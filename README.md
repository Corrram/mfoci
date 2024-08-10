# Factor selection with MFOCI

`mfoci` is a Python package designed for financial analysts and researchers who need to retrieve and analyze stock market data.
 This package implements the MFOCI algorithm for factor selection in financial data. Supported are also LASSO and k-FOCI factor selection methods as well as data retrieval from Fama-French and Yahoo Finance.

## Installation

Install `mfoci` using pip:

```bash
pip install mfoci
```


## Quick Usage

```python
from mfoci import get_fama_french_data, load_stock_returns
from mfoci import select_factors

# Fetch data
tickers = ["MSFT", "AAPL", "NVDA", "AMZN", "META", "GOOG"]
response_vars = load_stock_returns(tickers, start_date="2013-01-02", end_date="2024-01-01")
factors = get_fama_french_data("2013-01-03", end_date="2024-01-01")

# MFOCI factor selection
mfoci_selected, t_values = select_factors(factors, response_vars, "mfoci")
```


## Further usage

```python
from mfoci import get_fama_french_data, load_stock_returns, load_volatility_index
from mfoci import filter_for_common_indices, select_factors

# Fetch Fama-French factors
factors = get_fama_french_data("2004-01-01", end_date="2024-01-01")

# Load stock returns for specific tickers
tickers = ["MSFT", "AAPL", "NVDA", "AMZN", "META", "GOOG"]
response_vars = load_stock_returns(tickers, start_date="2013-01-01", end_date="2024-01-01")

# Load VIX data
response_vars = load_volatility_index("^VIX", start_date="2004-01-01", end_date="2024-01-01")

# Filter for common dates
factors, response_vars = filter_for_common_indices(factors, response_vars)

# Factor selection using LASSO
lasso_selected, coef = select_factors(factors, response_vars, "lasso")

# KFOCI factor selection (ensure Rscript is installed and path is set)
r_path = "C:/Program Files/R/R-4.3.3/bin/x64/Rscript"
kfoci_gauss_selected = select_factors(
    factors, response_vars, "kfoci", r_path=r_path, kernel="rbfdot"
)
kfoci_laplace_selected = select_factors(
    factors, response_vars, "kfoci", r_path=r_path, kernel="laplacedot"
)

# MFOCI factor selection
mfoci_selected, t_values = select_factors(factors, response_vars, "mfoci")
```
