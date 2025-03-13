import logging
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from mfoci.methods.multivar_chatterjee import xi_q_n_calculate


log = logging.getLogger(__name__)


def mfoci(
    factors: pd.DataFrame,
    response_vars: pd.DataFrame,
    report_insignificant: bool = False,
    shuffle: bool = True,
    max_iterations: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[int], List[float]]:
    """
    Multivariate FOCI (Feature Ordering by Conditional Independence)

    Implements a feature selection algorithm that ranks factors based on their
    predictive power for the response variables while controlling for previously
    selected factors.

    Algorithm steps:
    1. Start with an empty set of selected factors
    2. For each iteration, compute dependence measure for each remaining factor
    3. Select the factor with the highest dependence measure
    4. If the dependence measure decreases and report_insignificant=False, stop
    5. Otherwise, continue until all factors are selected

    Parameters:
    -----------
    factors : pd.DataFrame
        DataFrame containing the predictor variables
    response_vars : pd.DataFrame
        DataFrame containing the response variables
    report_insignificant : bool, default=False
        If True, continue selecting factors even after the dependence measure decreases
    shuffle : bool, default=True
        If True, try different orderings of response variables and average the results
    max_iterations : int, optional
        Maximum number of factors to select. If None, may select up to all factors
    verbose : bool, default=True
        If True, display progress bars and log messages

    Returns:
    --------
    selected_factors : List[int]
        List of column indices of selected factors in order of importance
    max_ts : List[float]
        List of dependence measures for each selected factor

    Notes:
    ------
    This implementation is based on the multivariate extension of FOCI algorithm.
    """
    # Validate inputs
    if factors.shape[0] != response_vars.shape[0]:
        raise ValueError("factors and response_vars must have the same number of rows")

    if not np.isfinite(factors.values).all():
        raise ValueError("factors contains non-finite values (NaN or Inf)")

    if not np.isfinite(response_vars.values).all():
        raise ValueError("response_vars contains non-finite values (NaN or Inf)")

    # Initialize variables
    q = response_vars.shape[1]  # Number of response variables
    p = factors.shape[1]  # Number of predictor variables

    if max_iterations is None:
        max_iterations = p
    else:
        max_iterations = min(max_iterations, p)

    selected_factors = []
    max_ts = []
    max_t = 0
    n_selected = 0

    # Determine the number of response permutations to try
    y_order_range = range(q) if shuffle else range(1)

    # Log start information
    if verbose:
        log.info(
            f"Starting MFOCI factor selection with {p} factors and {q} response variables. "
            f"Will select up to {max_iterations} factors."
        )

    # Main algorithm loop
    for i in range(max_iterations):
        if verbose:
            log.info(f"\nIteration {i + 1}:")

        # Calculate conditional dependence for each remaining factor
        t_js = _calculate_factor_scores(
            factors, response_vars, selected_factors, y_order_range, verbose
        )

        # Check stopping condition
        current_max_t = max(t_js)
        if current_max_t <= max_t and n_selected == 0:
            n_selected = i
            if not report_insignificant:
                break

        # Select the factor with highest score
        max_t = current_max_t
        argmax = t_js.index(max_t)
        selected_factors.append(argmax)
        max_ts.append(max_t)

    # If we never set n_selected (all iterations showed improvement or report_insignificant=True)
    if n_selected == 0:
        n_selected = len(selected_factors)

    # Log results
    if verbose:
        _log_results(factors, response_vars, selected_factors, max_ts, n_selected)

    return selected_factors, max_ts


def _calculate_factor_scores(
    factors: pd.DataFrame,
    response_vars: pd.DataFrame,
    selected_factors: List[int],
    y_order_range: range,
    verbose: bool = True,
) -> List[float]:
    """
    Calculate conditional dependence scores for each factor not yet selected.

    Parameters:
    -----------
    factors : pd.DataFrame
        DataFrame containing all predictor variables
    response_vars : pd.DataFrame
        DataFrame containing all response variables
    selected_factors : List[int]
        List of indices of factors already selected
    y_order_range : range
        Range of response variable permutations to try
    verbose : bool
        Whether to show progress bar

    Returns:
    --------
    t_js : List[float]
        List of conditional dependence scores for each factor
    """
    p = factors.shape[1]
    t_js = []

    # Use tqdm for progress reporting if verbose is True
    iterator = tqdm(range(p)) if verbose else range(p)

    for j in iterator:
        # Skip already selected factors
        if j in selected_factors:
            t_js.append(0)
            continue

        # Try different orderings of response variables
        t_ks = []
        for k in y_order_range:
            # Create a permutation of response variables
            shuffled_y = _shuffle_responses(response_vars, k)

            # Calculate conditional dependence
            candidate_factors = selected_factors + [j]
            t_k = xi_q_n_calculate(factors.iloc[:, candidate_factors], shuffled_y)
            t_ks.append(t_k)

        # Average the scores across different response orderings
        t_j = sum(t_ks) / len(y_order_range)
        t_js.append(t_j)

    return t_js


def _shuffle_responses(response_vars: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Create a permutation of response variables by rotating columns.

    Parameters:
    -----------
    response_vars : pd.DataFrame
        DataFrame containing response variables
    k : int
        Index to start the rotation

    Returns:
    --------
    shuffled_y : pd.DataFrame
        DataFrame with permuted response variables
    """
    return pd.concat([response_vars.iloc[:, k:], response_vars.iloc[:, :k]], axis=1)


def _log_results(
    factors: pd.DataFrame,
    response_vars: pd.DataFrame,
    selected_factors: List[int],
    max_ts: List[float],
    n_selected: int,
) -> None:
    """
    Log the results of the MFOCI algorithm.

    Parameters:
    -----------
    factors : pd.DataFrame
        DataFrame containing all predictor variables
    response_vars : pd.DataFrame
        DataFrame containing all response variables
    selected_factors : List[int]
        List of indices of selected factors
    max_ts : List[float]
        List of dependence measures for selected factors
    n_selected : int
        Number of significant factors selected
    """
    # Format the T scores
    t_formatted = [str(round(t, 3)) for t in max_ts]

    # Get the column names of selected factors
    selected_names = [factors.columns[i] for i in selected_factors[:n_selected]]

    # Log the results
    log.info("\nMFOCI results:")
    log.info(
        f"Predictive indicators for {', '.join(response_vars.columns)}"
        f" are (in this order) {', '.join(selected_names)}.\n"
        f"The corresponding T's are {' '.join(t_formatted)}.\n"
        f"Number of selected variables is {n_selected}."
    )
    print("Done!")
