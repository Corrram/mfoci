.. mfoci documentation master file, created by
   sphinx-quickstart on Sat Aug 10 10:31:52 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mfoci documentation
===================

**mfoci** is a package designed for model-free variable selection.

.. It accompanies the `Model-free variable selection for high-dimensional data <https://www.degruyter.com/document/doi/10.1515/demo-2024-0002/html>`_ article released in the Dependence Modeling journal.

** Basic usage**

.. code-block:: python

   from mfoci import get_fama_french_data, load_stock_returns
   from mfoci import select_factors

   # Fetch data
   tickers = ["MSFT", "AAPL", "NVDA", "AMZN", "META", "GOOG"]
   response_vars = load_stock_returns(tickers, start_date="2013-01-02", end_date="2024-01-01")
   factors = get_fama_french_data("2013-01-03", end_date="2024-01-01")

   # MFOCI factor selection
   mfoci_selected, t_values = select_factors(factors, response_vars, "mfoci")

**Further usage**

.. code-block:: python

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


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
