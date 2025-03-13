import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from mfoci.methods.multivar_chatterjee import (
    xi_q_n_calculate,
    t_y_fat_x,
    count_at_least_as_large,
    nearest_neighbor_indices,
)


class TestCountAtLeastAsLarge:
    def test_unique_values(self):
        """Test with an array of unique values."""
        y = np.array([1, 3, 5, 7, 9])
        expected = np.array([5, 4, 3, 2, 1])  # Number of elements >= each value
        result = count_at_least_as_large(y)
        assert_array_equal(result, expected)

    def test_duplicate_values(self):
        """Test with an array containing duplicate values."""
        y = np.array([1, 3, 3, 5, 7])
        expected = np.array([5, 4, 4, 2, 1])  # Number of elements >= each value
        result = count_at_least_as_large(y)
        assert_array_equal(result, expected)

    def test_single_value(self):
        """Test with a single element array."""
        y = np.array([42])
        expected = np.array([1])
        result = count_at_least_as_large(y)
        assert_array_equal(result, expected)

    def test_empty_array(self):
        """Test with an empty array."""
        y = np.array([])
        expected = np.array([])
        result = count_at_least_as_large(y)
        assert_array_equal(result, expected)


class TestNearestNeighborIndices:
    def test_basic_2d_array(self):
        """Test with a simple 2D array where nearest neighbors are clear."""
        data = np.array([[0, 0], [1, 0], [0, 1], [10, 10]])
        # Point 0 is closest to points 1 and 2 (both at distance 1), but implementation would pick the first one
        # Point 1 is closest to point 0 (distance 1)
        # Point 2 is closest to point 0 (distance 1)
        # Point 3 is far from all others, but closest to point 2
        expected = np.array([1, 0, 0, 1])
        result = nearest_neighbor_indices(data)
        assert_array_equal(result, expected)

    def test_identical_points(self):
        """Test with an array of identical points."""
        data = np.array([[1, 1], [1, 1], [1, 1]])
        # Each point should pick another point as its nearest neighbor
        # Since self-distance is set to infinity
        # Implementation would pick the first non-self index
        expected = np.array([1, 0, 0])
        result = nearest_neighbor_indices(data)
        assert_array_equal(result, expected)

    def test_different_dimensions(self):
        """Test with points in different dimensions."""
        data = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # All points are equidistant from point 0, but implementation would pick first one
        # Points 1, 2, 3 are all closest to point 0
        expected = np.array([1, 0, 0, 0])
        result = nearest_neighbor_indices(data)
        assert_array_equal(result, expected)


class TestTYFatX:
    def test_perfect_dependence(self):
        """Test with perfect deterministic relationship."""
        y = np.array(range(1, 100))
        x = y.reshape(-1, 1)
        result = t_y_fat_x(x, y)
        # Perfect dependence should give a value close to 1
        assert result > 0.9

    def test_independence(self):
        """Test with independent variables."""
        np.random.seed(42)  # For reproducibility
        n = 150
        x = np.random.normal(0, 1, (n, 2))
        y = np.random.normal(0, 1, n)  # Completely independent of x
        result = t_y_fat_x(x, y)
        # Independent variables should give a value close to 0
        assert result < 0.2

    def test_partial_dependence(self):
        """Test with partial dependence (noisy relationship)."""
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, (n, 1))
        y = x.flatten() + np.random.normal(0, 0.5, n)  # y = x + noise
        result = t_y_fat_x(x, y)
        # Partial dependence should give intermediate value
        assert 0.2 < result < 0.9

    def test_one_point(self):
        """Test with a single data point."""
        x = np.array([[42]])
        y = np.array([42])
        # With one point, nearest neighbor is itself (leading to undefined value)
        # Our implementation sets self-distance to infinity, so behavior is defined
        result = t_y_fat_x(x, y)
        assert np.isfinite(result)


class TestXiQNCalculate:
    one_to_100 = np.array(range(1, 100))

    def test_univariate_x_y(self):
        """Test with univariate x and y."""
        x = pd.DataFrame({"x": self.one_to_100})
        y = pd.DataFrame({"y": self.one_to_100})  # y = x (perfect dependence)
        result = xi_q_n_calculate(x, y)
        # Perfect dependence should give value close to 1
        assert result > 0.9

    def test_multivariate_x_univariate_y(self):
        """Test with multivariate x and univariate y."""
        x = pd.DataFrame({"x1": self.one_to_100, "x2": self.one_to_100[::-1]})
        y = pd.DataFrame({"y": self.one_to_100})  # y = x1 (perfect dependence with x1)
        result = xi_q_n_calculate(x, y)
        # Perfect dependence should give value close to 1
        assert result > 0.9

    def test_multivariate_x_y(self):
        """Test with multivariate x and multivariate y."""
        x = pd.DataFrame({"x1": self.one_to_100, "x2": self.one_to_100[::-1]})
        y = pd.DataFrame(
            {"y1": self.one_to_100, "y2": self.one_to_100[::-1]}  # y1 = x1  # y2 = x2
        )
        result = xi_q_n_calculate(x, y)
        # Perfect dependence should give value close to 1
        assert result > 0.9

    def test_independence(self):
        """Test with independent variables."""
        np.random.seed(42)
        n = 50
        x = pd.DataFrame(
            {"x1": np.random.normal(0, 1, n), "x2": np.random.normal(0, 1, n)}
        )
        y = pd.DataFrame(
            {"y1": np.random.normal(0, 1, n), "y2": np.random.normal(0, 1, n)}
        )
        result = xi_q_n_calculate(x, y)
        # Independent variables should give value close to 0
        assert result < 0.3

    def test_partial_dependence(self):
        """Test with partial dependence structure."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)

        # y1 depends on x1, y2 depends on x2, both with noise
        y1 = x1 + np.random.normal(0, 0.5, n)
        y2 = x2 + np.random.normal(0, 0.5, n)

        x = pd.DataFrame({"x1": x1, "x2": x2})
        y = pd.DataFrame({"y1": y1, "y2": y2})

        result = xi_q_n_calculate(x, y)
        # Partial dependence should give intermediate value
        assert 0.3 < result < 0.9

    def test_conditional_independence(self):
        """Test the conditional independence properties."""
        np.random.seed(42)
        n = 50

        # Create variables with conditional independence pattern
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)

        # y1 depends only on x1
        # y2 depends on y1 (and thus indirectly on x1) and on x2
        y1 = x1 + np.random.normal(0, 0.3, n)
        y2 = y1 + x2 + np.random.normal(0, 0.3, n)

        x = pd.DataFrame({"x1": x1, "x2": x2})
        y = pd.DataFrame({"y1": y1, "y2": y2})

        # Test that xi_q_n_calculate handles this conditional structure
        result = xi_q_n_calculate(x, y)
        assert 0.3 < result < 0.9
