import struct
import threading
import time
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import numpy as np
import pytest

try:
    from numcodecs.blosc import Blosc

    from numcodecs import blosc
except ImportError:  # pragma: no cover
    pytest.skip("numcodecs.blosc not available", allow_module_level=True)


from tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode,
    check_err_decode_object_buffer,
    check_err_encode_object_buffer,
    check_max_buffer_size,
)

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
    None,  # was snappy
    Blosc(shuffle=Blosc.SHUFFLE, blocksize=0),
    Blosc(shuffle=Blosc.SHUFFLE, blocksize=2**8),
    Blosc(cname='lz4', clevel=1, shuffle=Blosc.NOSHUFFLE, blocksize=2**8),
]
# compressors can be left out of a build (e.g. -Dblosc_zstd=disabled with c-blosc2)
codecs = [c if c is None or c.cname in blosc.list_compressors() else None for c in codecs]


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
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype='i8').view('M8[ns]'),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype='i8').view('m8[ns]'),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype='i8').view('M8[m]'),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype='i8').view('m8[m]'),
    np.empty(0, dtype='u1'),
]


def _skip_null(codec):
    if codec is None:
        pytest.skip("codec has been removed")


@pytest.fixture(scope='module', params=[True, False, None])
def use_threads(request):
    return request.param


@pytest.mark.parametrize('array', arrays)
@pytest.mark.parametrize('codec', codecs)
def test_encode_decode(array, codec):
    _skip_null(codec)
    check_encode_decode(array, codec)


def test_empty_encode_decode(use_threads):
    blosc.use_threads = use_threads
    try:
        check_encode_decode(np.empty(0, dtype='u1'), Blosc())
    finally:
        blosc.use_threads = None


def test_empty_decode_rejects_nonempty_destination():
    codec = Blosc()
    encoded = codec.encode(b'')
    with pytest.raises(RuntimeError, match='non-empty destination buffer'):
        codec.decode(encoded, out=bytearray(1))


def test_decompress_allows_trailing_bytes():
    codec = Blosc()
    original = b'some data to compress'
    encoded = codec.encode(original)
    assert codec.decode(encoded + b'padding') == original


def test_decompress_rejects_truncated_frame():
    codec = Blosc()
    encoded = codec.encode(b'some data to compress')
    with pytest.raises(RuntimeError, match='buffer is truncated'):
        codec.decode(encoded[:-1])


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
    actual = repr(Blosc(cname='blosclz', clevel=5, shuffle=Blosc.AUTOSHUFFLE, blocksize=1024))
    assert expect == actual


def test_eq():
    assert Blosc() == Blosc()
    assert Blosc(cname='lz4') != Blosc(cname='zstd')
    assert Blosc(clevel=1) != Blosc(clevel=9)
    assert Blosc(cname='lz4') != 'foo'


def test_compress_blocksize_default(use_threads):
    arr = np.arange(1000, dtype='i4')

    blosc.use_threads = use_threads

    # default blocksize
    enc = blosc.compress(arr, b'lz4', 1, Blosc.NOSHUFFLE)
    _, _, blocksize = blosc._cbuffer_sizes(enc)
    assert blocksize > 0

    # explicit default blocksize
    enc = blosc.compress(arr, b'lz4', 1, Blosc.NOSHUFFLE, 0)
    _, _, blocksize = blosc._cbuffer_sizes(enc)
    assert blocksize > 0


@pytest.mark.parametrize('bs', [2**7, 2**8])
def test_compress_blocksize(use_threads, bs):
    arr = np.arange(1000, dtype='i4')

    blosc.use_threads = use_threads

    enc = blosc.compress(arr, b'lz4', 1, Blosc.NOSHUFFLE, bs)
    _, _, blocksize = blosc._cbuffer_sizes(enc)
    assert blocksize == bs


def test_compress_complib(use_threads):
    arr = np.arange(1000, dtype='i4')
    expected_complibs = {
        'lz4': 'LZ4',
        'lz4hc': 'LZ4',
        'blosclz': 'BloscLZ',
        'zlib': 'Zlib',
        'zstd': 'Zstd',
    }
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


@pytest.mark.parametrize('dtype', ['i1', 'i2', 'i4', 'i8'])
def test_compress_metainfo(dtype, use_threads):
    arr = np.arange(1000, dtype=dtype)
    for shuffle in Blosc.NOSHUFFLE, Blosc.SHUFFLE, Blosc.BITSHUFFLE:
        blosc.use_threads = use_threads
        for cname in blosc.list_compressors():
            enc = blosc.compress(arr, cname.encode(), 1, shuffle)
            typesize, did_shuffle, _ = blosc._cbuffer_metainfo(enc)
            assert typesize == arr.dtype.itemsize
            if typesize == 1 and shuffle == Blosc.SHUFFLE and blosc.LIBRARY == 'c-blosc2':
                # c-blosc2 omits the byte shuffle flag when it is a no-op
                assert did_shuffle in (Blosc.NOSHUFFLE, shuffle)
            else:
                assert did_shuffle == shuffle


def test_compress_autoshuffle(use_threads):
    arr = np.arange(8000)
    for dtype in 'i1', 'i2', 'i4', 'i8', 'f2', 'f4', 'f8', 'bool', 'S10':
        varr = arr.view(dtype)
        blosc.use_threads = use_threads
        for cname in blosc.list_compressors():
            enc = blosc.compress(varr, cname.encode(), 1, Blosc.AUTOSHUFFLE)
            typesize, did_shuffle, _ = blosc._cbuffer_metainfo(enc)
            assert typesize == varr.dtype.itemsize
            if typesize == 1:
                assert did_shuffle == Blosc.BITSHUFFLE
            else:
                assert did_shuffle == Blosc.SHUFFLE


def test_config_blocksize():
    # N.B., we want to be backwards compatible with any config where blocksize is not
    # explicitly stated

    # blocksize not stated
    config = {"cname": 'lz4', "clevel": 1, "shuffle": Blosc.SHUFFLE}
    codec = Blosc.from_config(config)
    assert codec.blocksize == 0

    # blocksize stated
    config = {"cname": 'lz4', "clevel": 1, "shuffle": Blosc.SHUFFLE, "blocksize": 2**8}
    codec = Blosc.from_config(config)
    assert codec.blocksize == 2**8


def test_backwards_compatibility():
    check_backwards_compatibility(Blosc.codec_id, arrays, codecs)


def _encode_worker(data):
    cname = 'zlib' if 'zlib' in blosc.list_compressors() else 'lz4'
    compressor = Blosc(cname=cname, clevel=9, shuffle=Blosc.SHUFFLE)
    return compressor.encode(data)


def _decode_worker(enc):
    compressor = Blosc()
    return compressor.decode(enc)


@pytest.mark.parametrize('pool', [Pool, ThreadPool])
def test_multiprocessing(use_threads, pool):
    data = np.arange(1000000)
    enc = _encode_worker(data)

    pool = pool(5)

    try:
        blosc.use_threads = use_threads

        # test with process pool and thread pool

        # test encoding
        enc_results = pool.map(_encode_worker, [data] * 5)
        assert all(len(enc) == len(e) for e in enc_results)

        # test decoding
        dec_results = pool.map(_decode_worker, [enc] * 5)
        assert all(data.nbytes == len(d) for d in dec_results)

        # tidy up
        pool.close()
        pool.join()

    finally:
        blosc.use_threads = None  # restore default


def test_set_nthreads_races_global_context():
    # Regression test: blosc_set_nthreads re-creates Blosc's global context, so it must
    # be serialized against compress/decompress calls that use that context.
    data = np.arange(200_000, dtype='i4')
    stop = threading.Event()
    errors = []

    def worker():
        try:
            while not stop.is_set():
                enc = blosc.compress(data, b'lz4', 5)
                assert blosc.decompress(enc) == data.tobytes()
        except Exception as e:
            errors.append(e)

    def toggler():
        n = 1
        while not stop.is_set():
            n = 1 + (n % 4)
            blosc.set_nthreads(n)

    # force every thread onto the global-context path
    blosc.use_threads = True
    original_nthreads = blosc.get_nthreads()
    try:
        threads = [threading.Thread(target=worker) for _ in range(8)]
        threads.append(threading.Thread(target=toggler))
        for t in threads:
            t.start()
        time.sleep(1)
        stop.set()
        for t in threads:
            t.join()
    finally:
        blosc.use_threads = None  # restore default
        blosc.set_nthreads(original_nthreads)
    assert not errors


def test_err_decode_object_buffer():
    check_err_decode_object_buffer(Blosc())


def test_err_encode_object_buffer():
    check_err_encode_object_buffer(Blosc())


@pytest.mark.parametrize('codec', codecs)
def test_decompression_error_handling(codec):
    _skip_null(codec)
    with pytest.raises(RuntimeError):
        codec.decode(bytearray())
    with pytest.raises(RuntimeError):
        codec.decode(bytearray(0))


@pytest.mark.parametrize('codec', codecs)
def test_max_buffer_size(codec):
    _skip_null(codec)
    assert codec.max_buffer_size == 2**31 - 1
    check_max_buffer_size(codec)


def test_typesize_explicit():
    arr = np.arange(100).astype("int64")
    itemsize = arr.itemsize
    codec_no_type_size = Blosc(shuffle=Blosc.SHUFFLE)
    codec_itemsize = Blosc(shuffle=Blosc.SHUFFLE, typesize=itemsize)
    encoded_without_itemsize = codec_no_type_size.encode(arr.tobytes())
    encoded_with_itemsize = codec_itemsize.encode(arr.tobytes())
    # third byte encodes the `typesize`
    assert encoded_without_itemsize[3] == 1  # inferred from bytes i.e., 1
    assert encoded_with_itemsize[3] == itemsize  # given as a constructor argument


def test_typesize_less_than_1():
    with pytest.raises(ValueError, match=r"Cannot use typesize"):
        Blosc(shuffle=Blosc.SHUFFLE, typesize=0)
    compressor = Blosc(shuffle=Blosc.SHUFFLE)
    # not really something that should be done in practice, but good for testing.
    compressor._typesize = 0
    arr = np.arange(100)
    with pytest.raises(ValueError, match=r"Cannot use typesize"):
        compressor.encode(arr.tobytes())


@pytest.mark.parametrize(
    "shuffle", [Blosc.NOSHUFFLE, Blosc.SHUFFLE, Blosc.BITSHUFFLE, Blosc.AUTOSHUFFLE]
)
def test_zero_itemsize(shuffle):
    # a structured dtype with a zero-length field has itemsize 0, which blosc
    # rejects as a typesize (and older versions crashed on, #516)
    arr = np.zeros((5,), dtype=[("foo", int, (0,))])
    assert arr.dtype.itemsize == 0
    enc = Blosc(shuffle=shuffle).encode(arr)
    nbytes, _, _ = blosc._cbuffer_sizes(enc)
    assert nbytes == 0


def test_config_no_typesize():
    codec = Blosc(shuffle=Blosc.SHUFFLE, typesize=5)
    config = codec.get_config()
    assert "typesize" not in config


@pytest.mark.parametrize('shuffle', [Blosc.SHUFFLE, Blosc.BITSHUFFLE])
@pytest.mark.parametrize('blocksize', [0, 2**8, 4097])
@pytest.mark.parametrize('nbytes', [1, 31, 33, 127, 1000, 65537])
@pytest.mark.parametrize('typesize', [1, 2, 3, 4, 7, 8, 12, 16, 17, 32])
def test_shuffle_roundtrip_typesizes(typesize, nbytes, blocksize, shuffle):
    # Exercises the SIMD (un)shuffle kernels, including the scalar tail used when a
    # block is not a multiple of the vector width or the typesize.
    src = np.random.default_rng(0).integers(0, 64, nbytes, dtype='u1').tobytes()
    codec = Blosc(cname='lz4', clevel=5, shuffle=shuffle, blocksize=blocksize, typesize=typesize)
    assert bytes(codec.decode(codec.encode(src))) == src


@pytest.mark.parametrize('cname', blosc.list_compressors())
@pytest.mark.parametrize('shuffle', [Blosc.NOSHUFFLE, Blosc.SHUFFLE, Blosc.BITSHUFFLE])
@pytest.mark.parametrize('typesize', [1, 2, 3, 8])
def test_output_is_blosc1_format(cname, shuffle, typesize):
    # Chunks must stay readable by C-Blosc 1.x: format version 2 in the first byte
    src = np.arange(4096, dtype='u1').tobytes()
    enc = Blosc(cname=cname, shuffle=shuffle, typesize=typesize).encode(src)
    assert enc[0] == 2


def _blosc1_header(nbytes, cbytes, typesize=1, blocksize=1):
    # version 2, lz4, no flags, then nbytes, blocksize, cbytes (little endian)
    return struct.pack('<BBBBIII', 2, 1, 0, typesize, nbytes, blocksize, cbytes)


def test_decompress_rejects_short_buffer():
    with pytest.raises(RuntimeError, match='buffer is too small'):
        blosc.decompress(b'\x02\x01\x00')


def test_decompress_rejects_invalid_header():
    # a declared size above the maximum buffer size
    header = _blosc1_header(nbytes=2**31 - 1, cbytes=16)
    with pytest.raises(RuntimeError, match='header validation failed'):
        blosc.decompress(header)


def test_decompress_rejects_truncated_header_size():
    # the header says there are more compressed bytes than were given
    header = _blosc1_header(nbytes=10, cbytes=1000)
    with pytest.raises(RuntimeError, match='buffer is truncated'):
        blosc.decompress(header)


def _blosc2_special_chunk(version, flags, nbytes=1024):
    # a 32-byte chunk with the Blosc2 extended header (used by c-blosc2 whenever both
    # the shuffle and bitshuffle flag bits are set), marked as an uninitialized
    # special chunk: c-blosc2 would decode it without writing the output at all
    header = struct.pack('<BBBBiii', version, 1, flags, 1, nbytes, nbytes, 32)
    blosc2_flags = 0x4 << 4  # BLOSC2_SPECIAL_UNINIT
    return header + bytes(15) + bytes([blosc2_flags])


@pytest.mark.skipif(blosc.LIBRARY != 'c-blosc2', reason='c-blosc2 backend only')
@pytest.mark.parametrize(
    'chunk',
    [
        # Blosc2 format versions with a plain 16-byte header
        struct.pack('<BBBBiii', 3, 1, 0, 1, 16, 16, 32) + bytes(16),
        struct.pack('<BBBBiii', 6, 1, 0, 1, 16, 16, 32) + bytes(16),
        # extended (Blosc2) header with a Blosc2 or a Blosc1 format version
        _blosc2_special_chunk(version=5, flags=0x5),
        _blosc2_special_chunk(version=2, flags=0x5),
    ],
    ids=['v3', 'v6', 'v5-extended-uninit', 'v2-extended-uninit'],
)
def test_decompress_rejects_non_blosc1_format(chunk, use_threads):
    # this codec only writes Blosc1-format chunks and C-Blosc 1.x reads nothing else,
    # so the c-blosc2 backend must not decode other chunks either
    blosc.use_threads = use_threads
    try:
        with pytest.raises(RuntimeError, match='not a Blosc1-format frame'):
            blosc.decompress(chunk)
    finally:
        blosc.use_threads = None


def test_decompress_reads_only_within_frame(use_threads):
    # Point the first block at a copy of its own compressed data placed after the
    # frame. Bytes after the frame are allowed (and ignored), but must never be read
    # as part of it, whichever decompression path is taken.
    src = np.tile(np.arange(64, dtype='u1'), 2**12).tobytes()
    codec = Blosc(cname='lz4', clevel=5, shuffle=Blosc.NOSHUFFLE, blocksize=2**16)
    enc = bytearray(codec.encode(src))
    nbytes, cbytes, blocksize = blosc._cbuffer_sizes(enc)
    assert enc[0] == 2  # Blosc1 format
    assert not enc[2] & 0x2  # not memcpyed
    nblocks = nbytes // blocksize
    assert nblocks > 1
    bstarts = struct.unpack_from(f'<{nblocks}i', enc, 16)
    block0 = enc[bstarts[0] : bstarts[1]]
    struct.pack_into('<i', enc, 16, cbytes)
    tampered = bytes(enc) + bytes(block0)

    blosc.use_threads = use_threads
    try:
        assert bytes(blosc.decompress(bytes(codec.encode(src)) + bytes(block0))) == src
        with pytest.raises(RuntimeError):
            blosc.decompress(tampered)
    finally:
        blosc.use_threads = None


@pytest.mark.parametrize('blocksize', [10, 255, 10**6])
@pytest.mark.parametrize('typesize', [1, 3, 8])
def test_blocksize_larger_than_buffer_or_not_multiple_of_typesize(blocksize, typesize):
    # c-blosc 1.x adjusts the blocksize itself; c-blosc2 needs to be given a valid one
    src = np.arange(1000, dtype='u1').tobytes()
    for shuffle in (Blosc.NOSHUFFLE, Blosc.SHUFFLE, Blosc.BITSHUFFLE):
        codec = Blosc(cname='lz4', shuffle=shuffle, blocksize=blocksize, typesize=typesize)
        assert bytes(codec.decode(codec.encode(src))) == src


def test_library():
    assert blosc.LIBRARY in ('c-blosc', 'c-blosc2')
    assert blosc.VERSION_STRING


def test_compressors_listed():
    # these are built into Blosc whichever library it is built on
    assert {'blosclz', 'lz4', 'lz4hc'} <= set(blosc.list_compressors())
