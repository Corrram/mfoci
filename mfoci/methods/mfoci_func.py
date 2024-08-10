import logging

import pandas as pd
from tqdm import tqdm

from mfoci.methods.multivar_chatterjee import xi_q_n_calculate


log = logging.getLogger(__name__)


def mfoci(factors, response_vars, report_insignificant=False, shuffle=True):
    """
    Multivariate FOCI - implements slide 31 from Jonathan's presentation

    :param factors: pd.DataFrame
    :param response_vars: pd.DataFrame
    :param report_insignificant: bool
    :param shuffle: bool

    :return: list
    """
    q = response_vars.shape[1]
    p = factors.shape[1]
    selected_factors = []
    max_t = 0
    max_ts = []
    n_selected = 0
    log.info(
        f"Starting MFOCI factor selection with {p} factors and {q} response variables. "
        f"Depending on the number of selected factors "
        f"there can be up to {p} iterations."
    )
    y_order_range = range(q) if shuffle else range(1)
    for i in range(p):
        log.info(f"\nIteration {i + 1}:")
        t_js = []
        for j in tqdm(range(p)):
            if j in selected_factors:
                t_js.append(0)
                continue
            t_ks = []
            for k in y_order_range:  # shuffle y order to see effect on T^q
                shuffled_y = pd.concat(
                    [response_vars.iloc[:, k:], response_vars.iloc[:, :k]], axis=1
                )
                t_k = xi_q_n_calculate(
                    factors.iloc[:, selected_factors + [j]], shuffled_y
                )
                t_ks.append(t_k)
            t_j = sum(t_ks) / len(y_order_range)  # average over different orders of y
            t_js.append(t_j)
        if max(t_js) <= max_t and n_selected == 0:
            n_selected = i
            if not report_insignificant:
                break
        max_t = max(t_js)
        argmax = t_js.index(max_t)
        selected_factors.append(argmax)
        max_ts.append(max_t)

    t = [str(round(i, 3)) for i in max_ts]
    log.info("\nMFOCI results:")
    ind_str = ", ".join(factors.columns[selected_factors])
    log.info(
        f"Predictive indicators for {', '.join(response_vars.columns)}"
        f" are (in this order) {ind_str}.\n"
        f"The corresponding T's are {' '.join(t)}.\n"
        f"Number of selected variables is {n_selected}."
    )
    print("Done!")
    return selected_factors, max_ts
