import numpy as np
import pandas as pd
from scipy import stats


def xi_q_n_calculate(xvec, yvec):
    """
    Calculate the T^q value for given multivariate x and multivariate y.
    Implements the formula at the bottom of slide 21 in Jonathan's presentation, see
    also 2022 Ansari, Fuchs - A Simple Extension of T, formula (4).
    """
    xvec.reset_index(drop=True, inplace=True)
    yvec.reset_index(drop=True, inplace=True)
    q = yvec.shape[1]

    t_nom = 0
    t_den = 0
    for i in range(q):
        y = yvec.iloc[:, i].values
        prev_y = yvec.iloc[:, :i]
        x = pd.concat([xvec, prev_y], axis=1).values
        t_nom += t_y_fat_x(x, y)
        if i > 0:
            t_den += t_y_fat_x(prev_y.values, y)
    t = 1 - (q - t_nom) / (q - t_den)
    return t


def t_y_fat_x(xvec, y):
    """
    Calculate the T^q value for given multivariate x and univariate y.
    Implements the formula at the bottom of slide 17 in Jonathan's presentation.
    Corresponds to 2021 Azadkia, Chatterjee - A Simple Measure of Conditional Dependence
    the formula directly before Theorem 2.2 with xvec = Z and y = Y.
    """
    n = len(y)
    yrank = stats.rankdata(y, method="ordinal")
    # compute for each y the number of values that are at least as large
    # l_k = count_at_least_as_large(y)
    l_k = n - yrank + 1

    nn = nearest_neighbor_indices(xvec)
    # nn_old = nearest_neighbor_indices_old(xvec)
    min_yrank_nn = np.minimum(yrank, yrank[nn])
    nom = np.mean(n * min_yrank_nn - l_k**2)
    # den = np.mean(l_k * yrank)
    den = (n + 1) * (n + 2) / 6
    frac = nom / den
    return frac


def count_at_least_as_large(y):
    """
    For each element in the array, count the number of elements
    that are at least as large as the element.

    Parameters
    ----------
    y : numpy.ndarray
        1D array of numerical values.

    Returns
    -------
    counts : numpy.ndarray
        Array of counts for each element.
    """
    n = len(y)
    counts = np.zeros(n, dtype=int)

    for i in range(n):
        counts[i] = np.sum(y >= y[i])

    return counts


def nearest_neighbor_indices(data):
    """
    Find the index of the nearest neighbor for each row in the given 2D
    array using the euclidean metric.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)

    Returns
    -------
    nearest_indices : array, shape (n_samples,)
        The index of the nearest neighbor for each row.
    """
    n_samples = data.shape[0]
    nearest_indices = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # Compute the squared Euclidean distance from the i-th row to all other rows
        distances = np.sum((data - data[i]) ** 2, axis=1)
        distances[i] = np.inf  # Exclude self-distance
        nearest_indices[i] = np.argmin(distances)

    return nearest_indices
