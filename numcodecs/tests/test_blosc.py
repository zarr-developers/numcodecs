# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import itertools
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool


import numpy as np
import pytest


from numcodecs import blosc
from numcodecs.blosc import Blosc
from numcodecs.tests.common import (check_encode_decode, check_config,
                                    check_backwards_compatibility,
                                    check_err_decode_object_buffer,
                                    check_err_encode_object_buffer)


codecs = [
    Blosc(shuffle=Blosc.SHUFFLE),
    Blosc(clevel=0, shuffle=Blosc.SHUFFLE),
    Blosc(cname='lz4', shuffle=Blosc.SHUFFLE),
    Blosc(cname='lz4', clevel=1, shuffle=Blosc.NOSHUFFLE),
    Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE),
    Blosc(cname='lz4', clevel=9, shuffle=Blosc.BITSHUFFLE),
    Blosc(cname='zlib', clevel=1, shuffle=0),
    Blosc(cname='zstd', clevel=1, shuffle=1),
    Blosc(cname='blosclz', clevel=1, shuffle=2),
    Blosc(cname='snappy', clevel=1, shuffle=2),
    Blosc(shuffle=Blosc.SHUFFLE, blocksize=0),
    Blosc(shuffle=Blosc.SHUFFLE, blocksize=2**8),
    Blosc(cname='lz4', clevel=1, shuffle=Blosc.NOSHUFFLE, blocksize=2**8),
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
]


def test_encode_decode():
    for arr, codec in itertools.product(arrays, codecs):
        check_encode_decode(arr, codec)


def test_config():
    codec = Blosc(cname='zstd', clevel=3, shuffle=1)
    check_config(codec)
    codec = Blosc(cname='lz4', clevel=1, shuffle=2, blocksize=2**8)
    check_config(codec)


def test_repr():
    expect = "Blosc(cname='zstd', clevel=3, shuffle=SHUFFLE, blocksize=0)"
    actual = repr(Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE, blocksize=0))
    assert expect == actual
    expect = "Blosc(cname='lz4', clevel=1, shuffle=NOSHUFFLE, blocksize=256)"
    actual = repr(Blosc(cname='lz4', clevel=1, shuffle=Blosc.NOSHUFFLE, blocksize=256))
    assert expect == actual
    expect = "Blosc(cname='zlib', clevel=9, shuffle=BITSHUFFLE, blocksize=512)"
    actual = repr(Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE, blocksize=512))
    assert expect == actual
    expect = "Blosc(cname='blosclz', clevel=5, shuffle=AUTOSHUFFLE, blocksize=1024)"
    actual = repr(Blosc(cname='blosclz', clevel=5, shuffle=Blosc.AUTOSHUFFLE,
                        blocksize=1024))
    assert expect == actual


def test_eq():
    assert Blosc() == Blosc()
    assert Blosc(cname='lz4') != Blosc(cname='zstd')
    assert Blosc(clevel=1) != Blosc(clevel=9)
    assert Blosc(cname='lz4') != 'foo'


def test_compress_blocksize():
    arr = np.arange(1000, dtype='i4')

    for use_threads in True, False, None:
        blosc.use_threads = use_threads

        # default blocksize
        enc = blosc.compress(arr, b'lz4', 1, Blosc.NOSHUFFLE)
        _, _, blocksize = blosc.cbuffer_sizes(enc)
        assert blocksize > 0

        # explicit default blocksize
        enc = blosc.compress(arr, b'lz4', 1, Blosc.NOSHUFFLE, 0)
        _, _, blocksize = blosc.cbuffer_sizes(enc)
        assert blocksize > 0

        # custom blocksize
        for bs in 2**7, 2**8:
            enc = blosc.compress(arr, b'lz4', 1, Blosc.NOSHUFFLE, bs)
            _, _, blocksize = blosc.cbuffer_sizes(enc)
            assert blocksize == bs


def test_compress_complib():
    arr = np.arange(1000, dtype='i4')
    expected_complibs = {
        'lz4': 'LZ4',
        'lz4hc': 'LZ4',
        'blosclz': 'BloscLZ',
        'snappy': 'Snappy',
        'zlib': 'Zlib',
        'zstd': 'Zstd',
    }
    for use_threads in True, False, None:
        blosc.use_threads = use_threads
        for cname in blosc.list_compressors():
            enc = blosc.compress(arr, cname.encode(), 1, Blosc.NOSHUFFLE)
            complib = blosc.cbuffer_complib(enc)
            expected_complib = expected_complibs[cname]
            assert complib == expected_complib
        with pytest.raises(ValueError):
            # capitalized cname
            blosc.compress(arr, b'LZ4', 1)
        with pytest.raises(ValueError):
            # bad cname
            blosc.compress(arr, b'foo', 1)


def test_compress_metainfo():
    for dtype in 'i1', 'i2', 'i4', 'i8':
        arr = np.arange(1000, dtype=dtype)
        for shuffle in Blosc.NOSHUFFLE, Blosc.SHUFFLE, Blosc.BITSHUFFLE:
            for use_threads in True, False, None:
                blosc.use_threads = use_threads
                for cname in blosc.list_compressors():
                    enc = blosc.compress(arr, cname.encode(), 1, shuffle)
                    typesize, did_shuffle, _ = blosc.cbuffer_metainfo(enc)
                    assert typesize == arr.dtype.itemsize
                    assert did_shuffle == shuffle


def test_compress_autoshuffle():
    arr = np.arange(8000)
    for dtype in 'i1', 'i2', 'i4', 'i8', 'f2', 'f4', 'f8', 'bool', 'S10':
        varr = arr.view(dtype)
        for use_threads in True, False, None:
            blosc.use_threads = use_threads
            for cname in blosc.list_compressors():
                enc = blosc.compress(varr, cname.encode(), 1, Blosc.AUTOSHUFFLE)
                typesize, did_shuffle, _ = blosc.cbuffer_metainfo(enc)
                assert typesize == varr.dtype.itemsize
                if typesize == 1:
                    assert did_shuffle == Blosc.BITSHUFFLE
                else:
                    assert did_shuffle == Blosc.SHUFFLE


def test_config_blocksize():
    # N.B., we want to be backwards compatible with any config where blocksize is not
    # explicitly stated

    # blocksize not stated
    config = dict(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    codec = Blosc.from_config(config)
    assert codec.blocksize == 0

    # blocksize stated
    config = dict(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE, blocksize=2**8)
    codec = Blosc.from_config(config)
    assert codec.blocksize == 2**8


def test_backwards_compatibility():
    check_backwards_compatibility(Blosc.codec_id, arrays, codecs)


def _encode_worker(data):
    compressor = Blosc(cname='zlib', clevel=9, shuffle=Blosc.SHUFFLE)
    enc = compressor.encode(data)
    return enc


def _decode_worker(enc):
    compressor = Blosc()
    data = compressor.decode(enc)
    return data


def test_multiprocessing():
    data = np.arange(1000000)
    enc = _encode_worker(data)

    try:
        for use_threads in None, True, False:
            blosc.use_threads = use_threads

            # test with process pool and thread pool
            for pool in Pool(5), ThreadPool(5):

                # test encoding
                enc_results = pool.map(_encode_worker, [data] * 5)
                assert all([len(enc) == len(e) for e in enc_results])

                # test decoding
                dec_results = pool.map(_decode_worker, [enc] * 5)
                assert all([data.nbytes == len(d) for d in dec_results])

                # tidy up
                pool.close()
                pool.join()

    finally:
        blosc.use_threads = None  # restore default


def test_err_decode_object_buffer():
    check_err_decode_object_buffer(Blosc())


def test_err_encode_object_buffer():
    check_err_encode_object_buffer(Blosc())
