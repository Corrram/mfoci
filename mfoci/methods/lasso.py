import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, MultiTaskLassoCV


def select_indicators_with_lasso(factors, response_vars):
    """
    Select indicators using Lasso regression.

    Parameters:
    factors (pd.DataFrame): Factors/predictors dataframe
    response_vars (pd.DataFrame): Response variables dataframe

    Returns:
    tuple: (selected_columns, coefficients)
    """
    # Store column names
    response_column_names = response_vars.columns.tolist()

    # Determine if univariate or multivariate
    y_univariate = response_vars.shape[1] == 1

    if y_univariate:
        # Extract the first column as the target
        response_data = response_vars.iloc[:, 0].values
        # Use LassoCV for univariate case
        lasso = LassoCV(cv=5)
        lasso.fit(factors.values, response_data)
        coef = lasso.coef_
        # Take absolute value for feature importance
        abs_coef = np.abs(coef)
    else:
        # Use MultiTaskLassoCV for multivariate case
        response_data = response_vars.values
        lasso = MultiTaskLassoCV(cv=5, random_state=0, max_iter=10000)
        lasso.fit(factors.values, response_data)
        coef = lasso.coef_
        # Sum absolute values across all target variables
        abs_coef = np.sum(np.abs(coef), axis=0)

    # Sort by importance (absolute coefficient value)
    sorted_indices = np.argsort(abs_coef)[::-1]  # Descending order

    # For univariate: select columns with positive coefficients
    # For multivariate: select columns with positive sum of absolute coefficients
    if y_univariate:
        # Select features with positive coefficients
        mask = coef > 0
        selected_indices = np.where(mask)[0]
    else:
        # Here we check if any coefficient for a feature is positive across all responses
        mask = np.any(coef > 0, axis=0)
        selected_indices = np.where(mask)[0]

    # Sort selected indices by their importance
    sorted_selected = sorted(
        [i for i in selected_indices], key=lambda i: abs_coef[i], reverse=True
    )

    selected_cols = factors.columns[sorted_selected]

    # Print results
    print("Lasso results:")
    print(
        f"Predictive indicators for {', '.join(response_column_names)} "
        f"are (in this order) {', '.join(selected_cols)}"
    )
    print(f"Number of selected variables is {len(selected_cols)}.")
    rounded_coef = np.round(abs_coef[sorted_indices], 3)
    print(f"Average absolute coefficient per indicator is {rounded_coef}.\n")

    return selected_cols, coef
