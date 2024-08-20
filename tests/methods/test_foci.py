import copul
import numpy as np
import pandas as pd
import pytest
import logging

from mfoci import codec

log = logging.getLogger(__name__)


def test_codec():
    """
    This test ensures consistency with the R implementation of the codec function
    and the xi function from the xicorpy package.
    """
    Y = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    Z = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    result = codec(Y, Z)
    assert result == 0.25


observations = [10_000, 15_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000]


@pytest.mark.parametrize("n_obs", observations)
def test_xi_estimation_of_amh(n_obs):
    # true xi = 0.0352...
    family = copul.Families.ALI_MIKHAIL_HAQ
    param = 0.6
    log.info(f"Number of observations: {n_obs}")
    xi = compute_xi_from_data(family, param, n_obs)
    assert 0.02 <= xi <= 0.05


@pytest.mark.parametrize("n_obs", observations)
def test_xi_estimation_of_clayton(n_obs):
    family = copul.Families.CLAYTON
    param = 2  # true xi ~ 0.3
    log.info(f"Number of observations: {n_obs}")
    xi = compute_xi_from_data(family, param, n_obs)
    assert 0.2 <= xi <= 0.4


def compute_xi_from_data(family, param, n_obs=20_000):
    np.random.seed(121)
    copula = family.value(theta=param)
    log.info(f"Family: {family}, parameter: {param}")
    data = copula.rvs(n_obs)
    df = pd.DataFrame(data, columns=["x", "z"])
    xi = codec(df["x"], df["z"])
    return xi


def test_xi_estimation_for_perfect_dependence():
    np.random.seed(121)
    n = 10_000
    factor_cols = ["F1", "F2"]
    factors = pd.DataFrame(np.random.normal(size=(n, 2)), columns=factor_cols)
    stocks = pd.DataFrame(factors["F1"] + factors["F2"], columns=["A"])
    estimated_xi = codec(stocks, factors)["A"]
    assert estimated_xi > 0.95


def test_xi_estimation_for_independence():
    np.random.seed(121)
    n = 1_000
    factor_cols = ["F1", "F2"]
    factors = pd.DataFrame(np.random.normal(size=(n, 2)), columns=factor_cols)
    stocks = pd.DataFrame(np.random.normal(size=n), columns=["A"])
    estimated_xi = codec(stocks, factors)["A"]
    assert estimated_xi < 0.05


def test_xi_estimation_for_almost_perfect_dependence():
    np.random.seed(121)
    matr = np.eye(500)
    ccop = copul.CheckerboardCopula(matr)
    data = ccop.rvs(5_000)
    df = pd.DataFrame(data, columns=["x", "z"])
    estimated_xi = codec(df["x"], df["z"])
    assert estimated_xi > 0.9
