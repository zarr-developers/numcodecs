# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import itertools


import numpy as np


from numcodecs import blosc
from numcodecs.blosc import Blosc
from numcodecs.tests.common import check_encode_decode, check_config, check_backwards_compatibility


codecs = [
    Blosc(),
    Blosc(clevel=0),
    Blosc(cname='lz4'),
    Blosc(cname='lz4', clevel=1, shuffle=Blosc.NOSHUFFLE),
    Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE),
    Blosc(cname='lz4', clevel=9, shuffle=Blosc.BITSHUFFLE),
    Blosc(cname='zlib', clevel=1, shuffle=0),
    Blosc(cname='zstd', clevel=1, shuffle=1),
    Blosc(cname='blosclz', clevel=1, shuffle=2),
    Blosc(cname='snappy', clevel=1, shuffle=2),
    Blosc(blocksize=0),
    Blosc(blocksize=2**8),
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
    np.random.choice([b'a', b'bb', b'ccc'], size=1000).reshape(10, 10, 10)
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
        for bs in 2**7, 2**7:
            enc = blosc.compress(arr, b'lz4', 1, Blosc.NOSHUFFLE, bs)
            _, _, blocksize = blosc.cbuffer_sizes(enc)
        assert blocksize == bs


def test_config_blocksize():
    # N.B., we want to be backwards compatible with any config where blocksize is not explicitly
    # stated

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
