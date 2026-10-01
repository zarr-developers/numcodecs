import numpy as np
import pytest

from numcodecs import LZ4, Blosc, Zstd, blosc
from numcodecs.compat import ensure_contiguous_ndarray

temporal_dtypes = [
    np.dtype([('date', 'M8[D]'), ('value', 'i8')]),
    np.dtype([('duration', '>m8[ns]'), ('value', '<i2')]),
    np.dtype([('nested', [('date', '>M8[ms]'), ('duration', '<m8[us]')]), ('tag', 'S3')]),
    np.dtype([('nested', [('dates', 'M8[D]', (2,)), ('value', '>i4')], (2,))]),
    np.dtype(
        {'names': ['date', 'value'], 'formats': ['M8[D]', 'u1'], 'offsets': [1, 12], 'itemsize': 16}
    ),
]
codecs = [Zstd(), LZ4(), Blosc(), Blosc(shuffle=Blosc.BITSHUFFLE)]


def make_array(dtype, shape, order):
    arr = np.zeros(shape, dtype=dtype, order=order)
    # Fill padding too, so raw-byte comparisons detect accidental field-wise copies.
    raw = arr.reshape(-1, order='A').view('u1')
    raw[:] = np.arange(raw.size, dtype='u1')
    return arr


@pytest.mark.parametrize('dtype', temporal_dtypes)
@pytest.mark.parametrize('shape', [(), (0,), (2, 3)])
@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('flatten', [False, True])
@pytest.mark.parametrize('writeable', [False, True])
def test_structured_temporal_buffer_view(dtype, shape, order, flatten, writeable):
    arr = make_array(dtype, shape, order)
    arr.setflags(write=writeable)
    expected = arr.tobytes(order='A')
    result = ensure_contiguous_ndarray(arr, flatten=flatten)

    assert memoryview(result).tobytes(order='A') == expected
    assert result.dtype.itemsize == dtype.itemsize
    assert result.shape == ((arr.size,) if flatten else shape)
    if not flatten:
        assert result.strides == arr.strides
    assert result.ctypes.data == arr.ctypes.data
    assert result.flags.writeable == writeable
    assert arr.dtype == dtype
    assert arr.tobytes(order='A') == expected


@pytest.mark.parametrize('codec', codecs)
@pytest.mark.parametrize('dtype', temporal_dtypes)
@pytest.mark.parametrize('shape', [(), (2, 3)])
@pytest.mark.parametrize('order', ['C', 'F'])
def test_structured_temporal_roundtrip(codec, dtype, shape, order):
    arr = make_array(dtype, shape, order)
    expected = arr.tobytes(order='A')
    arr.setflags(write=False)
    encoded = codec.encode(arr)
    assert codec.decode(encoded) == expected
    if isinstance(codec, Blosc):
        typesize, _, _ = blosc._cbuffer_metainfo(encoded)
        assert typesize == dtype.itemsize

    # An unaligned buffer is still contiguous and must be written without a copy.
    backing = bytearray(arr.nbytes + 2)
    out = np.ndarray(shape, dtype=dtype, buffer=backing, offset=1, order=order)
    assert codec.decode(memoryview(encoded), out=out) is out
    assert out.dtype == dtype
    assert out.shape == shape
    assert out.tobytes(order='A') == expected
    assert backing[0] == backing[-1] == 0
    assert arr.dtype == dtype
    assert arr.tobytes(order='A') == expected

    # Byte-format memoryviews remain valid inputs and outputs.
    assert codec.decode(codec.encode(memoryview(expected))) == expected
    raw_out = memoryview(bytearray(arr.nbytes))
    assert codec.decode(encoded, out=raw_out) is raw_out
    assert raw_out.tobytes() == expected


@pytest.mark.parametrize('codec', codecs)
def test_issue_755(codec):
    arr = np.array(
        [('19000101', 123), ('19021201', 321)], dtype=[('date', 'datetime64[D]'), ('num', 'i8')]
    )
    out = np.empty_like(arr)
    assert codec.decode(codec.encode(arr), out=out) is out
    np.testing.assert_array_equal(out, arr)


@pytest.mark.parametrize('codec', codecs)
@pytest.mark.parametrize('dtype', [np.dtype('i8'), temporal_dtypes[0]])
@pytest.mark.parametrize('as_memoryview', [False, True])
def test_decode_readonly_output(codec, dtype, as_memoryview):
    arr = make_array(dtype, (3,), 'C')
    encoded = codec.encode(arr.tobytes())
    out = np.zeros_like(arr)
    expected = out.tobytes()
    out.setflags(write=False)
    if as_memoryview:
        out = memoryview(out.view(f'V{out.dtype.itemsize}'))
    with pytest.raises(ValueError, match='read-only'):
        codec.decode(encoded, out=out)
    assert out.tobytes() == expected


@pytest.mark.parametrize(
    'dtype',
    [
        np.dtype([('object', 'O')]),
        np.dtype([('nested', [('object', 'O'), ('date', 'M8[D]')])]),
        np.dtype([('objects', 'O', (2,))]),
    ],
)
@pytest.mark.parametrize('codec', codecs)
def test_structured_object_buffers_rejected(codec, dtype):
    arr = np.zeros(3, dtype=dtype)
    with pytest.raises(TypeError, match='object arrays are not supported'):
        codec.encode(arr)
    encoded = codec.encode(bytes(arr.nbytes))
    with pytest.raises(TypeError, match='object arrays are not supported'):
        codec.decode(encoded, out=arr)


@pytest.mark.parametrize('codec', codecs)
def test_structured_temporal_buffer_guards(codec):
    arr = make_array(temporal_dtypes[0], (6,), 'C')
    with pytest.raises(ValueError, match='contiguous'):
        codec.encode(arr[::2])
    encoded = codec.encode(arr.tobytes())
    with pytest.raises(ValueError, match='contiguous'):
        codec.decode(encoded, out=np.zeros(12, dtype=arr.dtype)[::2])
    with pytest.raises(ValueError, match='destination buffer too small'):
        codec.decode(encoded, out=np.zeros(5, dtype=arr.dtype))
    with pytest.raises(ValueError, match='Codec does not support buffers'):
        ensure_contiguous_ndarray(arr, max_buffer_size=arr.nbytes - 1)


def test_regular_structured_buffer_unchanged():
    dtype = np.dtype([('nested', [('value', '>i8')]), ('tag', 'S3')])
    arr = make_array(dtype, (2, 3), 'F')
    result = ensure_contiguous_ndarray(arr, flatten=False)
    assert result.dtype == dtype
    assert result is arr
    assert ensure_contiguous_ndarray(memoryview(arr), flatten=False).dtype == dtype
