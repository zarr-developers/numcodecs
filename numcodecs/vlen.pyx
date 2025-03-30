# cython: embedsignature=True
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# cython: language_level=3


cimport cython

from libc.stdint cimport uint8_t
from libc.string cimport memcpy

from cpython.buffer cimport PyBuffer_IsContiguous
from cpython.bytearray cimport (
    PyByteArray_AS_STRING,
    PyByteArray_FromStringAndSize,
)
from cpython.bytes cimport (
    PyBytes_AS_STRING,
    PyBytes_GET_SIZE,
    PyBytes_Check,
    PyBytes_FromStringAndSize,
)
from cpython.memoryview cimport PyMemoryView_GET_BUFFER
from cpython.unicode cimport (
    PyUnicode_AsUTF8String,
    PyUnicode_Check,
    PyUnicode_FromStringAndSize,
)

from numpy cimport ndarray

from .compat_ext cimport ensure_continguous_memoryview
from ._utils cimport store_le32, load_le32

import numpy as np

from .abc import Codec
from .compat import ensure_contiguous_ndarray


# 4 bytes to store number of items
cdef Py_ssize_t HEADER_LENGTH = 4


def check_out_param(out, n_items):
    if not isinstance(out, np.ndarray):
        raise TypeError('out must be 1-dimensional array')
    if out.dtype != object:
        raise ValueError('out must be object array')
    out = out.reshape(-1, order='A')
    if out.shape[0] < n_items:
        raise ValueError('out is too small')
    return out


class VLenUTF8(Codec):
    """Encode variable-length unicode string objects via UTF-8.

    Examples
    --------
    >>> import numcodecs
    >>> import numpy as np
    >>> x = np.array(['foo', 'bar', 'baz'], dtype='object')
    >>> codec = numcodecs.VLenUTF8()
    >>> codec.decode(codec.encode(x))
    array(['foo', 'bar', 'baz'], dtype=object)

    See Also
    --------
    numcodecs.pickles.Pickle, numcodecs.json.JSON, numcodecs.msgpacks.MsgPack

    Notes
    -----
    The encoded bytes values for each string are packed into a parquet-style byte array.

    """

    codec_id = 'vlen-utf8'

    def __init__(self):
        pass

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def encode(self, buf):
        cdef:
            Py_ssize_t i, l, n_items, data_length, total_length
            ndarray[object, ndim=1] input_values
            object[:] encoded_values
            int[:] encoded_lengths
            char* encv
            bytes b
            bytearray out
            char* data
            object u

        # normalise input
        input_values = np.asarray(buf, dtype=object).reshape(-1, order='A')

        # determine number of items
        n_items = input_values.shape[0]

        # setup intermediates
        encoded_values = np.empty(n_items, dtype=object)
        encoded_lengths = np.empty(n_items, dtype=np.intc)

        # first iteration to convert to bytes
        data_length = 0
        for i in range(n_items):
            u = input_values[i]
            if u is None or u == 0:  # treat these as missing value, normalize
                u = ''
            elif not PyUnicode_Check(u):
                raise TypeError('expected unicode string, found %r' % u)
            b = PyUnicode_AsUTF8String(u)
            l = PyBytes_GET_SIZE(b)
            encoded_values[i] = b
            data_length += l + 4  # 4 bytes to store item length
            encoded_lengths[i] = l

        # setup output
        total_length = HEADER_LENGTH + data_length
        out = PyByteArray_FromStringAndSize(NULL, total_length)

        # write header
        data = PyByteArray_AS_STRING(out)
        store_le32(<uint8_t*>data, n_items)

        # second iteration, store data
        data += HEADER_LENGTH
        for i in range(n_items):
            l = encoded_lengths[i]
            store_le32(<uint8_t*>data, l)
            data += 4
            encv = PyBytes_AS_STRING(encoded_values[i])
            memcpy(data, encv, l)
            data += l

        return out

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def decode(self, buf, out=None):
        cdef:
            memoryview buf_mv
            const Py_buffer* buf_pb
            const char* data
            const char* data_end
            Py_ssize_t i, l, n_items, data_length

        # obtain memoryview
        buf = ensure_contiguous_ndarray(buf)
        buf_mv = memoryview(buf)
        buf_pb = PyMemoryView_GET_BUFFER(buf_mv)

        # sanity checks
        if not PyBuffer_IsContiguous(buf_pb, b'A'):
            raise BufferError("`buf` must contain contiguous memory")
        if buf_pb.len < HEADER_LENGTH:
            raise ValueError('corrupt buffer, missing or truncated header')

        # obtain input data pointer
        data = <const char*>buf_pb.buf
        data_end = data + buf_pb.len

        # load number of items
        n_items = load_le32(<uint8_t*>data)

        # setup output
        if out is not None:
            out = check_out_param(out, n_items)
        else:
            out = np.empty(n_items, dtype=object)

        # iterate and decode - N.B., do not try to cast `out` as object[:]
        # as this causes segfaults, possibly similar to
        # https://github.com/cython/cython/issues/1608
        data += HEADER_LENGTH
        for i in range(n_items):
            if data + 4 > data_end:
                raise ValueError('corrupt buffer, data seem truncated')
            l = load_le32(<uint8_t*>data)
            data += 4
            if data + l > data_end:
                raise ValueError('corrupt buffer, data seem truncated')
            out[i] = PyUnicode_FromStringAndSize(data, l)
            data += l

        return out


class VLenBytes(Codec):
    """Encode variable-length byte string objects.

    Examples
    --------
    >>> import numcodecs
    >>> import numpy as np
    >>> x = np.array([b'foo', b'bar', b'baz'], dtype='object')
    >>> codec = numcodecs.VLenBytes()
    >>> codec.decode(codec.encode(x))
    array([b'foo', b'bar', b'baz'], dtype=object)

    See Also
    --------
    numcodecs.pickles.Pickle

    Notes
    -----
    The bytes values for each string are packed into a parquet-style byte array.

    """

    codec_id = 'vlen-bytes'

    def __init__(self):
        pass

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def encode(self, buf):
        cdef:
            Py_ssize_t i, l, n_items, data_length, total_length
            object[:] values
            int[:] lengths
            char* encv
            object b
            bytearray out
            char* data

        # normalise input
        values = np.asarray(buf, dtype=object).reshape(-1, order='A')

        # determine number of items
        n_items = values.shape[0]

        # setup intermediates
        lengths = np.empty(n_items, dtype=np.intc)

        # first iteration to find lengths
        data_length = 0
        for i in range(n_items):
            b = values[i]
            if b is None or b == 0:  # treat these as missing value, normalize
                b = b''
            elif not PyBytes_Check(b):
                raise TypeError('expected byte string, found %r' % b)
            l = PyBytes_GET_SIZE(b)
            data_length += l + 4  # 4 bytes to store item length
            lengths[i] = l

        # setup output
        total_length = HEADER_LENGTH + data_length
        out = PyByteArray_FromStringAndSize(NULL, total_length)

        # write header
        data = PyByteArray_AS_STRING(out)
        store_le32(<uint8_t*>data, n_items)

        # second iteration, store data
        data += HEADER_LENGTH
        for i in range(n_items):
            l = lengths[i]
            store_le32(<uint8_t*>data, l)
            data += 4
            encv = PyBytes_AS_STRING(values[i])
            memcpy(data, encv, l)
            data += l

        return out

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def decode(self, buf, out=None):
        cdef:
            memoryview buf_mv
            const Py_buffer* buf_pb
            const char* data
            const char* data_end
            Py_ssize_t i, l, n_items, data_length

        # obtain memoryview
        buf = ensure_contiguous_ndarray(buf)
        buf_mv = memoryview(buf)
        buf_pb = PyMemoryView_GET_BUFFER(buf_mv)

        # sanity checks
        if not PyBuffer_IsContiguous(buf_pb, b'A'):
            raise BufferError("`buf` must contain contiguous memory")
        if buf_pb.len < HEADER_LENGTH:
            raise ValueError('corrupt buffer, missing or truncated header')

        # obtain input data pointer
        data = <const char*>buf_pb.buf
        data_end = data + buf_pb.len

        # load number of items
        n_items = load_le32(<uint8_t*>data)

        # setup output
        if out is not None:
            out = check_out_param(out, n_items)
        else:
            out = np.empty(n_items, dtype=object)

        # iterate and decode - N.B., do not try to cast `out` as object[:]
        # as this causes segfaults, possibly similar to
        # https://github.com/cython/cython/issues/1608
        data += HEADER_LENGTH
        for i in range(n_items):
            if data + 4 > data_end:
                raise ValueError('corrupt buffer, data seem truncated')
            l = load_le32(<uint8_t*>data)
            data += 4
            if data + l > data_end:
                raise ValueError('corrupt buffer, data seem truncated')
            out[i] = PyBytes_FromStringAndSize(data, l)
            data += l

        return out


class VLenArray(Codec):
    """Encode variable-length 1-dimensional arrays via UTF-8.

    Examples
    --------
    >>> import numcodecs
    >>> import numpy as np
    >>> x1 = np.array([1, 3, 5], dtype=np.int16)
    >>> x2 = np.array([4], dtype=np.int16)
    >>> x3 = np.array([7, 9], dtype=np.int16)
    >>> x = np.array([x1, x2, x3], dtype='object')
    >>> codec = numcodecs.VLenArray('<i2')
    >>> codec.decode(codec.encode(x))
    array([array([1, 3, 5], dtype=int16), array([4], dtype=int16),
           array([7, 9], dtype=int16)], dtype=object)

    See Also
    --------
    numcodecs.pickles.Pickle, numcodecs.json.JSON, numcodecs.msgpacks.MsgPack

    Notes
    -----
    The binary data for each array are packed into a parquet-style byte array.

    """

    codec_id = 'vlen-array'

    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)

    def get_config(self):
        config = dict()
        config['id'] = self.codec_id
        config['dtype'] = self.dtype.str
        return config

    def __repr__(self):
        return '%s(dtype=%r)' % (type(self).__name__, self.dtype.str,)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def encode(self, buf):
        cdef:
            Py_ssize_t i, l, n_items, data_length, total_length
            object[:] values
            object[:] normed_values
            int[:] lengths
            const char* encv
            bytes b
            bytearray out
            char* data
            memoryview value_mv
            const Py_buffer* value_pb
            object v

        # normalise input
        values = np.asarray(buf, dtype=object).reshape(-1, order='A')

        # determine number of items
        n_items = values.shape[0]

        # setup intermediates
        normed_values = np.empty(n_items, dtype=object)
        lengths = np.empty(n_items, dtype=np.intc)

        # first iteration to convert to bytes
        data_length = 0
        for i in range(n_items):
            v = values[i]
            if v is None:
                v = np.array([], dtype=self.dtype)
            else:
                v = np.ascontiguousarray(v, self.dtype)
            if v.ndim != 1:
                raise ValueError('only 1-dimensional arrays are supported')
            l = v.nbytes
            normed_values[i] = v
            data_length += l + 4  # 4 bytes to store item length
            lengths[i] = l

        # setup output
        total_length = HEADER_LENGTH + data_length
        out = PyByteArray_FromStringAndSize(NULL, total_length)

        # write header
        data = PyByteArray_AS_STRING(out)
        store_le32(<uint8_t*>data, n_items)

        # second iteration, store data
        data += HEADER_LENGTH
        for i in range(n_items):
            l = lengths[i]
            store_le32(<uint8_t*>data, l)
            data += 4

            value_mv = ensure_continguous_memoryview(normed_values[i])
            value_pb = PyMemoryView_GET_BUFFER(value_mv)
            encv = <const char*>value_pb.buf

            memcpy(data, encv, l)
            data += l

        return out

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def decode(self, buf, out=None):
        cdef:
            memoryview buf_mv
            const Py_buffer* buf_pb
            const char* data
            const char* data_end
            object v
            memoryview v_mv
            Py_buffer* v_pb
            Py_ssize_t i, l, n_items, data_length

        # obtain memoryview
        buf = ensure_contiguous_ndarray(buf)
        buf_mv = memoryview(buf)
        buf_pb = PyMemoryView_GET_BUFFER(buf_mv)

        # sanity checks
        if not PyBuffer_IsContiguous(buf_pb, b'A'):
            raise BufferError("`buf` must contain contiguous memory")
        if buf_pb.len < HEADER_LENGTH:
            raise ValueError('corrupt buffer, missing or truncated header')

        # obtain input data pointer
        data = <const char*>buf_pb.buf
        data_end = data + buf_pb.len

        # load number of items
        n_items = load_le32(<uint8_t*>data)

        # setup output
        if out is not None:
            out = check_out_param(out, n_items)
        else:
            out = np.empty(n_items, dtype=object)

        # iterate and decode - N.B., do not try to cast `out` as object[:]
        # as this causes segfaults, possibly similar to
        # https://github.com/cython/cython/issues/1608
        data += HEADER_LENGTH
        for i in range(n_items):
            if data + 4 > data_end:
                raise ValueError('corrupt buffer, data seem truncated')
            l = load_le32(<uint8_t*>data)
            data += 4
            if data + l > data_end:
                raise ValueError('corrupt buffer, data seem truncated')

            # Create & fill array value
            v = np.empty((l,), dtype="uint8").view(self.dtype)
            v_mv = memoryview(v)
            v_pb = PyMemoryView_GET_BUFFER(v_mv)
            memcpy(v_pb.buf, data, l)

            out[i] = v
            data += l

        return out
