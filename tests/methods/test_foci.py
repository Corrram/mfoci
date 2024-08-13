import mfoci


def test_codec():
    """
    This test ensures consistency with the R implementation of the codec function
    and the xi function from the xicorpy package.
    """
    Y = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    Z = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    result = mfoci.codec(Y, Z)
    assert result == 0.25
