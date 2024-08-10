import numpy as np
from sklearn.linear_model import LassoCV, MultiTaskLassoCV


def select_indicators_with_lasso(factors, response_vars):
    y_univariate = response_vars.shape[1] == 1
    if y_univariate:
        response_vars = response_vars.iloc[:, 0]
        lasso = LassoCV(cv=5)
        # lasso = Lasso()
    else:
        lasso = MultiTaskLassoCV(cv=5, random_state=0, max_iter=10000)
    lasso.fit(factors.values, response_vars.values)
    coef = lasso.coef_
    if y_univariate:
        coef = np.abs(coef)
    else:
        coef = np.sum(np.abs(coef), axis=0)
    # sort ff5 columns by the sum of the absolute values of the coefficients
    sorted_coef = np.argsort(coef)[::-1]
    n_selected = np.sum(coef > 0)
    selected_cols = factors.iloc[:, sorted_coef].columns[:n_selected]
    print("Lasso results:")
    print(
        f"Predictive indicators for {', '.join(response_vars.columns)} "
        f"are (in this order) {', '.join(selected_cols)}"
    )
    print(f"Number of selected variables is {n_selected}.")
    rounded_coef = coef[sorted_coef].round(3)
    print(f"Average absolute coefficient per indicator is {rounded_coef}.\n")
    return selected_cols, coef
