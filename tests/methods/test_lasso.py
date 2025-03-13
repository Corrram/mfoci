import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# This is the critical part - patch the correct path where the function is defined
# Make sure this import path matches where the function is actually defined
from mfoci.methods.lasso import select_indicators_with_lasso


class TestLassoSelection:
    @pytest.fixture
    def mock_factors(self):
        # Create mock factor data
        data = {
            "factor1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "factor2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "factor3": [2.0, 2.0, 2.0, 2.0, 2.0],
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_univariate_response(self):
        # Create mock univariate response data
        data = {"response": [1.0, 2.0, 3.0, 2.0, 1.0]}
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_multivariate_response(self):
        # Create mock multivariate response data
        data = {
            "response1": [1.0, 2.0, 3.0, 2.0, 1.0],
            "response2": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
        return pd.DataFrame(data)

    @patch("builtins.print")  # Mock print to avoid output during tests
    def test_univariate_model_selection(
        self, mock_print, mock_factors, mock_univariate_response
    ):
        """Test that LassoCV is used for univariate response."""
        # The key is to patch the module where the function imports from
        with patch("mfoci.methods.lasso.LassoCV") as mock_lasso_cv:
            # Configure the mock
            instance = mock_lasso_cv.return_value
            instance.coef_ = np.array([0.1, 0.2, 0.3])

            # Call the function
            select_indicators_with_lasso(mock_factors, mock_univariate_response)

            # Assert that LassoCV was instantiated with expected parameters
            mock_lasso_cv.assert_called_once_with(cv=5)

            # Assert that fit was called with expected arguments
            instance.fit.assert_called_once()

    @patch("builtins.print")
    def test_multivariate_model_selection(
        self, mock_print, mock_factors, mock_multivariate_response
    ):
        """Test that MultiTaskLassoCV is used for multivariate response."""
        with patch("mfoci.methods.lasso.MultiTaskLassoCV") as mock_multi_lasso_cv:
            # Configure the mock
            instance = mock_multi_lasso_cv.return_value
            instance.coef_ = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

            # Call the function
            select_indicators_with_lasso(mock_factors, mock_multivariate_response)

            # Assert that MultiTaskLassoCV was instantiated with expected parameters
            mock_multi_lasso_cv.assert_called_once_with(
                cv=5, random_state=0, max_iter=10000
            )

            # Assert that fit was called with expected arguments
            instance.fit.assert_called_once()

    @patch("builtins.print")
    def test_univariate_selection_results(
        self, mock_print, mock_factors, mock_univariate_response
    ):
        """Test the selection results for univariate response."""
        with patch("mfoci.methods.lasso.LassoCV") as mock_lasso_cv:
            # Set up coefficients with different signs and magnitudes
            instance = mock_lasso_cv.return_value
            instance.coef_ = np.array([0.1, -0.2, 0.3])

            # Call the function
            selected_cols, coef = select_indicators_with_lasso(
                mock_factors, mock_univariate_response
            )

            # Check the returned results
            assert len(selected_cols) == 2  # Two positive coefficients
            assert selected_cols[0] == "factor3"  # Highest absolute coefficient
            assert selected_cols[1] == "factor1"  # Other positive coefficient

            # Check that coefficient array is absolute values and sorted
            np.testing.assert_allclose(coef, np.array([0.1, -0.2, 0.3]), rtol=1e-5)

    @patch("builtins.print")
    def test_multivariate_selection_results(
        self, mock_print, mock_factors, mock_multivariate_response
    ):
        """Test the selection results for multivariate response."""
        with patch("mfoci.methods.lasso.MultiTaskLassoCV") as mock_multi_lasso_cv:
            # Set up coefficients for two responses
            instance = mock_multi_lasso_cv.return_value
            instance.coef_ = np.array(
                [[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]]  # Response 1  # Response 2
            )

            # Call the function
            selected_cols, coef = select_indicators_with_lasso(
                mock_factors, mock_multivariate_response
            )

            # Check the returned results
            assert (
                len(selected_cols) == 3
            )  # All factors have positive coefficients in at least one response
            # Factors should be sorted by sum of absolute coefficients: factor3, factor2, factor1
            assert list(selected_cols) == ["factor3", "factor2", "factor1"]

    @patch("builtins.print")
    def test_no_selected_variables(
        self, mock_print, mock_factors, mock_univariate_response
    ):
        """Test case where no variables are selected (all coefficients <= 0)."""
        with patch("mfoci.methods.lasso.LassoCV") as mock_lasso_cv:
            # Configure the mock to return all non-positive coefficients
            instance = mock_lasso_cv.return_value
            instance.coef_ = np.array([-0.1, -0.2, 0.0])

            # Call the function
            selected_cols, coef = select_indicators_with_lasso(
                mock_factors, mock_univariate_response
            )

            # Check that no columns are selected
            assert len(selected_cols) == 0

    @patch("builtins.print")
    def test_print_output_format(
        self, mock_print, mock_factors, mock_univariate_response
    ):
        """Test that the function prints expected information."""
        with patch("mfoci.methods.lasso.LassoCV") as mock_lasso_cv:
            # Configure the mock
            instance = mock_lasso_cv.return_value
            instance.coef_ = np.array([0.1, 0.2, 0.0])

            # Call the function
            select_indicators_with_lasso(mock_factors, mock_univariate_response)

            # Assert that print was called with expected messages
            assert mock_print.call_count >= 3
            calls = [call[0][0] for call in mock_print.call_args_list]

            # Check that the expected information is printed
            assert any("Lasso results" in call for call in calls)
            assert any("Predictive indicators" in call for call in calls)
            assert any("Number of selected variables" in call for call in calls)
            assert any("Average absolute coefficient" in call for call in calls)
