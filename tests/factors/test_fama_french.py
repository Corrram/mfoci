import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
import io
from zipfile import ZipFile

# Import the module to test
from mfoci.factors.fama_french import get_fama_french_data

# Sample mock data in the format Kenneth French provides
MOCK_CSV_CONTENT = """5 Factors (2x3)
Average Value Weighted Returns -- Daily
January 1963 - December 2020
Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,0.39,-0.16,-0.25,-0.18,-0.31,0.010
19630702,0.11,0.12,0.05,0.04,0.09,0.010
19630703,0.11,-0.20,-0.05,-0.01,-0.09,0.010
19630705,0.90,0.08,0.15,0.11,0.10,0.010
"""


@pytest.fixture
def mock_response():
    """Create a mock response for requests"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    # Create compressed bytes (mimicking a zip file)
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('F-F_Research_Data_5_Factors_2x3_daily.CSV', MOCK_CSV_CONTENT)

    mock_resp.content = zip_buffer.getvalue()
    return mock_resp


def test_get_fama_french_data_response(mock_response):
    """Test that get_fama_french_data handles response correctly"""
    with patch('requests.get', return_value=mock_response):
        df = get_fama_french_data(start_date="1963-7-1", end_date="1963-7-5")

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 4 days of data

        # Check columns
        assert "Mkt-RF" in df.columns
        assert "SMB" in df.columns
        assert "HML" in df.columns
        assert "PLA-Unif" in df.columns
        assert "PLA-Gauss" in df.columns
        assert "PLA-Exp" in df.columns
        assert "RF" not in df.columns  # Should be removed

        # Check random columns
        assert all((df["PLA-Unif"] >= 0) & (df["PLA-Unif"] <= 1))  # Uniform distribution

        # Check index
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index[0] == pd.Timestamp('1963-07-01')


def test_get_fama_french_data_date_filtering(mock_response):
    """Test date filtering in get_fama_french_data"""
    with patch('requests.get', return_value=mock_response):
        # Test with limited date range
        df = get_fama_french_data(start_date="1963-7-2", end_date="1963-7-3")

        # Check filtering worked properly
        assert len(df) == 2
        assert df.index.min() == pd.Timestamp('1963-07-02')
        assert df.index.max() == pd.Timestamp('1963-07-03')


def test_get_fama_french_data_error_handling():
    """Test error handling in get_fama_french_data"""
    # Test HTTP error
    with patch('requests.get') as mock_get:
        mock_get.return_value = MagicMock()
        mock_get.return_value.status_code = 404

        with pytest.raises(ValueError, match="Failed to download data"):
            get_fama_french_data()

    # Test parsing error with invalid zip file
    with patch('requests.get') as mock_get:
        mock_get.return_value = MagicMock()
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b'Invalid content'

        with pytest.raises(ValueError, match="Failed to parse data"):
            get_fama_french_data()
