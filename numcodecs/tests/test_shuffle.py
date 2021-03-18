from multiprocessing import Pool
from multiprocessing.pool import ThreadPool


import numpy as np
import pytest


try:
    from numcodecs.shuffle import Shuffle
except ImportError:  # pragma: no cover
    pytest.skip(
        "numcodecs.shuffle not available", allow_module_level=True
    )


from numcodecs.tests.common import (check_encode_decode,
                                    check_config,
                                    check_backwards_compatibility)


codecs = [
    Shuffle(),
    Shuffle(elementsize=0),
    Shuffle(elementsize=4),
    Shuffle(elementsize=8)
]


# mix of dtypes: integer, float, bool, string
# mix of shapes: 1D, 2D, 3D
# mix of orders: C, F
arrays = [
    np.arange(1000, dtype='i4'),
    np.linspace(1000, 1001, 1000, dtype='f8'),
    np.random.normal(loc=1000, scale=1, size=(100, 10)),
    np.random.randint(0, 2, size=1000, dtype=bool).reshape(100, 10, order='F'),
    np.random.choice([b'a', b'bb', b'ccc'], size=1000).reshape(10, 10, 10),
    np.random.randint(0, 2**60, size=1000, dtype='u8').view('M8[ns]'),
    np.random.randint(0, 2**60, size=1000, dtype='u8').view('m8[ns]'),
    np.random.randint(0, 2**25, size=1000, dtype='u8').view('M8[m]'),
    np.random.randint(0, 2**25, size=1000, dtype='u8').view('m8[m]'),
    np.random.randint(-2**63, -2**63 + 20, size=1000, dtype='i8').view('M8[ns]'),
    np.random.randint(-2**63, -2**63 + 20, size=1000, dtype='i8').view('m8[ns]'),
    np.random.randint(-2**63, -2**63 + 20, size=1000, dtype='i8').view('M8[m]'),
    np.random.randint(-2**63, -2**63 + 20, size=1000, dtype='i8').view('m8[m]'),
]


@pytest.mark.parametrize('array', arrays)
@pytest.mark.parametrize('codec', codecs)
def test_encode_decode(array, codec):
    check_encode_decode(array, codec)


def test_config():
    codec = Shuffle()
    check_config(codec)
    codec = Shuffle(elementsize=8)
    check_config(codec)


def test_repr():
    expect = "Shuffle(elementsize=0)"
    actual = repr(Shuffle(elementsize=0))
    assert expect == actual
    expect = "Shuffle(elementsize=4)"
    actual = repr(Shuffle(elementsize=4))
    assert expect == actual
    expect = "Shuffle(elementsize=8)"
    actual = repr(Shuffle(elementsize=8))
    assert expect == actual
    expect = "Shuffle(elementsize=16)"
    actual = repr(Shuffle(elementsize=16))
    assert expect == actual


def test_eq():
    assert Shuffle() == Shuffle()
    assert Shuffle(elementsize=16) != Shuffle()


def _encode_worker(data):
    compressor = Shuffle()
    enc = compressor.encode(data)
    return enc


def _decode_worker(enc):
    compressor = Shuffle()
    data = compressor.decode(enc)
    return data


@pytest.mark.parametrize('pool', (Pool, ThreadPool))
def test_multiprocessing(pool):
    data = np.arange(1000000)
    enc = _encode_worker(data)

    pool = pool(5)

    # test with process pool and thread pool

    # test encoding
    enc_results = pool.map(_encode_worker, [data] * 5)
    assert all([len(enc) == len(e) for e in enc_results])

    # test decoding
    dec_results = pool.map(_decode_worker, [enc] * 5)
    assert all([data.nbytes == len(d) for d in dec_results])

    # tidy up
    pool.close()
    pool.join()


def test_backwards_compatibility():
    check_backwards_compatibility(Shuffle.codec_id, arrays, codecs)


# def test_err_decode_object_buffer():
#     check_err_decode_object_buffer(Shuffle())


# def test_err_encode_object_buffer():
#     check_err_encode_object_buffer(Shuffle())

# def test_decompression_error_handling():
#     for codec in codecs:
#         with pytest.raises(RuntimeError):
#             codec.decode(bytearray())
#         with pytest.raises(RuntimeError):
#             codec.decode(bytearray(0))

def test_expected_result():
    # Each byte of the 4 byte uint64 is shuffled in such a way
    # that for an array of length 4, the last byte of the last
    # element becomes the first byte of the first element
    # therefore [0, 0, 0, 1] becomes [2**((len-1)*8), 0, 0, 0]
    # (where 8 = bits in a byte)
    arr = np.array([0, 0, 0, 1], dtype='uint64')
    codec = Shuffle(elementsize=arr.data.itemsize)
    enc = codec.encode(arr)
    assert np.frombuffer(enc.data, arr.dtype)[0] == 2**((len(arr)-1)*8)


def test_incompatible_elementsize():
    with pytest.raises(ValueError):
        arr = np.arange(1001, dtype='u1')
        codec = Shuffle(elementsize=4)
        codec.encode(arr)
