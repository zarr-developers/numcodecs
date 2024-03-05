import numpy as np

import pytest

from numcodecs.bitinfo import BitInfo, exponent_bias, mutual_information


def test_bitinfo_initialization():
    bitinfo = BitInfo(0.5)
    assert bitinfo.info_level == 0.5
    assert bitinfo.axes is None

    bitinfo = BitInfo(0.5, axes=1)
    assert bitinfo.axes == [1]

    bitinfo = BitInfo(0.5, axes=[1, 2])
    assert bitinfo.axes == [1, 2]

    with pytest.raises(ValueError):
        BitInfo(-0.1)

    with pytest.raises(ValueError):
        BitInfo(1.1)

    with pytest.raises(ValueError):
        BitInfo(0.5, axes=1.5)

    with pytest.raises(ValueError):
        BitInfo(0.5, axes=[1, 1.5])


def test_bitinfo_encode():
    bitinfo = BitInfo(info_level=0.5)
    a = np.array([1.0, 2.0, 3.0], dtype="float32")
    encoded = bitinfo.encode(a)
    decoded = bitinfo.decode(encoded)
    assert decoded.dtype == a.dtype


def test_bitinfo_encode_errors():
    bitinfo = BitInfo(0.5)
    a = np.array([1, 2, 3], dtype="int32")
    with pytest.raises(TypeError):
        bitinfo.encode(a)

    a = np.array([1.0, 2.0, 3.0], dtype="float128")
    with pytest.raises(TypeError):
        bitinfo.encode(a)


def test_exponent_bias():
    assert exponent_bias("f2") == 15
    assert exponent_bias("f4") == 127
    assert exponent_bias("f8") == 1023

    with pytest.raises(ValueError):
        exponent_bias("int32")


def test_mutual_information():
    """ Test mutual information calculation

    Tests for changes to the mutual_information
    but not the correcteness of the original.
    """
    a = np.arange(10.0, dtype='float32')
    b = a + 1000
    c = a[::-1].copy()
    dt = np.dtype('uint32')
    a, b, c = map(lambda x: x.view(dt), [a, b, c])

    assert mutual_information(a, a).sum() == 7.020411549771797
    assert mutual_information(a, b).sum() == 0.0
    assert mutual_information(a, c).sum() == 0.6545015579460758
