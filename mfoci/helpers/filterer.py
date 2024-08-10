import logging

log = logging.getLogger(__name__)


def filter_for_common_indices(factors, response_vars):
    common_idx = response_vars.index.intersection(factors.index)
    filtered_response_vars = response_vars.loc[common_idx]
    filtered_factors = factors.loc[common_idx]
    n_response_dropped = response_vars.shape[0] - filtered_response_vars.shape[0]
    n_factors_dropped = factors.shape[0] - filtered_factors.shape[0]
    if n_response_dropped > 0:
        log.warning(f"Dropped {n_response_dropped} rows from response variables.")
    if n_factors_dropped > 0:
        log.warning(f"Dropped {n_factors_dropped} rows from factors.")
    log.info(f"Response variables shape: {filtered_response_vars.shape}")
    return filtered_factors, filtered_response_vars
