import decimal

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata


# Helper functions for codec
# noinspection PyPep8Naming
def codec(Y, Z, X=None, na_rm=True):
    """
    The conditional dependence coefficient (CODEC) is a measure of the amount of
    conditional dependence between a random variable Y and a random vector Z given
    a random vector X, based on an i.i.d. sample of (Y, Z, X).
    The coefficient is asymptotically guaranteed to be between 0 and 1.

    This implementation translates the FOCI.codec R method to Python.

    Parameters
    ----------
    Y : array-like
        The response variable.
    Z : array-like
        The conditioning variable.
    X : array-like, optional
        The conditioning variable. If None, the unconditional CODEC is calculated.
    na_rm : bool, optional
        A boolean value indicating whether to remove NAs. The default is True.

    Returns
    -------
    float
        The conditional dependence coefficient.

    Raises
    ------
    ValueError
        If the number of rows of Y, X, and Z are not equal.
        If the number of rows with no NAs is less than 2.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from mfoci import codec
    >>> n = 1000
    >>> x = np.random.rand(n, 2)
    >>> y = (x[:, 0] + x[:, 1]) % 1
    >>> y_2 = np.random.rand(n)
    >>> x_1_reshaped = x[:, 1].reshape(-1, 1)
    >>> x_0_reshaped = x[:, 0].reshape(-1, 1)
    >>> codec_y_x = codec(y, x)
    >>> z = np.random.randn(n)
    >>> z_reshaped = z.reshape(-1, 1)
    >>> codec_y_z_x = codec(y, z_reshaped, x)
    >>> codec_y_z = codec(y, z_reshaped)
    """
    if X is None:
        if isinstance(Z, list):
            Z = np.array(Z)
        if len(np.shape(Z)) == 1:
            Z = Z.reshape(-1, 1)
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)
        if not isinstance(Z, np.ndarray):
            Z = np.array(Z)
        if len(Y) != Z.shape[0]:
            raise ValueError("Number of rows of Y and Z should be equal.")
        if na_rm:
            mask = np.isfinite(Y) & np.all(np.isfinite(Z), axis=1)
            Z = Z[mask, :]
            Y = Y[mask]

        if len(Y) < 2:
            raise ValueError("Number of rows with no NAs should be bigger than 1.")

        return estimateT(Y, Z)

    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(Z, np.ndarray):
        Z = np.array(Z)
    if len(Y) != X.shape[0] or len(Y) != Z.shape[0] or Z.shape[0] != X.shape[0]:
        raise ValueError("Number of rows of Y, X, and Z should be equal.")
    if na_rm:
        mask = (
            np.isfinite(Y)
            & np.all(np.isfinite(Z), axis=1)
            & np.all(np.isfinite(X), axis=1)
        )
        Z = Z[mask, :]
        Y = Y[mask]
        X = X[mask, :]

    if len(Y) < 2:
        raise ValueError("Number of rows with no NAs should be bigger than 1.")

    return estimateConditionalT(Y, Z, X)


# noinspection PyPep8Naming
def estimateConditionalQ(Y, X, Z):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(Z, np.ndarray):
        Z = np.array(Z)

    n = len(Y)
    W = np.hstack((X, Z))

    nn_X = cKDTree(X).query(X, k=3)[1][:, 1]
    repeat_data = np.where(np.linalg.norm(X - X[nn_X], axis=1) == 0)[0]
    nn_index_X = handle_repeats(nn_X, repeat_data, X)

    nn_W = cKDTree(W).query(W, k=3)[1][:, 1]
    repeat_data_W = np.where(np.linalg.norm(W - W[nn_W], axis=1) == 0)[0]
    nn_index_W = handle_repeats(nn_W, repeat_data_W, W)

    R_Y = rankdata(Y, method="max")
    minimum_1 = np.minimum(R_Y, R_Y[nn_index_W])
    minimum_2 = np.minimum(R_Y, R_Y[nn_index_X])
    Q_n = np.sum(minimum_1 - minimum_2) / n**2
    return Q_n


# noinspection PyPep8Naming
def estimateConditionalS(Y, X):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    n = len(Y)

    nn_X = cKDTree(X).query(X, k=3)[1][:, 1]
    repeat_data = np.where(np.linalg.norm(X - X[nn_X], axis=1) == 0)[0]
    nn_index_X = handle_repeats(nn_X, repeat_data, X)

    R_Y = rankdata(Y, method="max")
    S_n = np.sum(R_Y - np.minimum(R_Y, R_Y[nn_index_X])) / n**2
    return S_n


# noinspection PyPep8Naming
def estimateConditionalT(Y, Z, X):
    S = estimateConditionalS(Y, X)
    if S == 0:
        return 1
    else:
        q = estimateConditionalQ(Y, X, Z)
        return q / S


# noinspection PyPep8Naming
def estimateQ(Y, X):
    # Convert X to a numpy array if it is not already
    X = np.array(X)

    n = len(Y)
    # Use cKDTree for nearest neighbor search
    tree = cKDTree(X)
    distances, nn_indices = tree.query(X, k=3)

    # Remove the first nearest neighbor for each x, which is x itself
    nn_index_X = nn_indices[:, 1]

    # Find all data points that are not unique
    repeat_data = np.where(distances[:, 1] == 0)[0]

    # Create a DataFrame to manage repeated data
    df_X = pd.DataFrame({"id": repeat_data, "group": nn_indices[repeat_data, 0]})

    # Function to select a random nearest neighbor
    def random_nn(ids):
        if len(ids) > 0:
            return np.random.choice(ids)
        return None

    df_X["rnn"] = df_X.groupby("group")["id"].transform(random_nn)
    nn_index_X[repeat_data] = df_X["rnn"].to_numpy()

    # Nearest neighbors with ties
    ties = np.where(distances[:, 1] == distances[:, 2])[0]
    ties = np.setdiff1d(ties, repeat_data)

    def helper_ties(a):
        a_point = X[a, :].reshape(1, -1)
        rest_points = np.delete(X, a, axis=0)
        rest_indices = np.delete(np.arange(len(X)), a)

        distances_to_others = np.linalg.norm(rest_points - a_point, axis=1)
        min_indices = np.where(distances_to_others == distances_to_others.min())[0]

        # Adjust indices since we removed one point
        adjusted_indices = rest_indices[min_indices]
        random_choice = np.random.choice(adjusted_indices)
        return random_choice

    if len(ties) > 0:
        tie_choice = [helper_ties(a) for a in ties]
        nn_index_X[ties] = np.array(tie_choice)

    R_Y = rankdata(Y, method="max")
    L_Y = rankdata(-Y, method="max")
    L_Y_dec = np.array([decimal.Decimal(float(val)) for val in L_Y])
    R_Y_dec = np.array([decimal.Decimal(float(val)) for val in R_Y])
    R_Y_nn = R_Y_dec[nn_index_X]  # R_Y at nearest neighbor indices
    min_values = np.minimum(R_Y_dec, R_Y_nn)  # Element-wise minimum
    L_Y_squared = L_Y_dec**2
    Q_n = np.mean(min_values - L_Y_squared / n) / n
    return float(Q_n)


# noinspection PyPep8Naming
def estimateS(Y):
    n = len(Y)
    L_Y = rankdata(-Y, method="max")
    L_Y_decimal = np.array([decimal.Decimal(float(val)) for val in L_Y])
    n_decimal = decimal.Decimal(n)

    S_n = np.sum(L_Y_decimal * (n_decimal - L_Y_decimal)) / (n_decimal**3)
    return float(S_n)


# noinspection PyPep8Naming
def estimateT(Y, X):
    S = estimateS(Y)
    if S == 0:
        return 1
    else:
        q = estimateQ(Y, X)
        return q / S


# noinspection PyPep8Naming
def handle_repeats(nn_index, repeat_data, X):
    if len(repeat_data) > 0:
        for i in repeat_data:
            unique_indices = np.where(np.linalg.norm(X[i] - X, axis=1) != 0)[0]
            nn_index[i] = np.random.choice(unique_indices)
    return nn_index


# Example usage
if __name__ == "__main__":
    n = 1000
    x = np.random.rand(n, 2)
    y = (x[:, 0] + x[:, 1]) % 1
    y_2 = np.random.rand(n)
    x_1_reshaped = x[:, 1].reshape(-1, 1)
    x_0_reshaped = x[:, 0].reshape(-1, 1)
    # print(codec(y, x_1_reshaped, x_0_reshaped))
    codec_y_x = codec(y, x)
    codec_y_2_x = codec(y_2, x)
    z = np.random.randn(n)
    z_reshaped = z.reshape(-1, 1)
    # print(codec(y, x, z_reshaped))
    # print(codec(y, z_reshaped, x))
