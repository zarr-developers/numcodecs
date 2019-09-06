# -*- coding: utf-8 -*-
# cython: embedsignature=True
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# cython: language_level=2
from __future__ import absolute_import, print_function, division


import cython
cimport cython
import numpy as np
from .abc import Codec
from .compat_ext cimport Buffer
from .compat_ext import Buffer
from .compat import ensure_contiguous_ndarray
from cpython cimport (PyBytes_GET_SIZE, PyBytes_AS_STRING, PyBytes_Check,
                      PyBytes_FromStringAndSize, PyUnicode_AsUTF8String)
from cpython.buffer cimport PyBUF_ANY_CONTIGUOUS
from libc.string cimport memcpy


cdef extern from "stdint_compat.h":
    void store_le32(char *c, int y)
    int load_le32(const char *c)


cdef extern from "Python.h":
    bytearray PyByteArray_FromStringAndSize(char *v, Py_ssize_t l)
    char* PyByteArray_AS_STRING(object string)
    object PyUnicode_FromStringAndSize(const char *u, Py_ssize_t size)
    int PyUnicode_Check(object text)


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


class VLenNDArray(Codec):
    """Encode variable-length n-dimensional arrays via UTF-8.

    Examples
    --------
    >>> import numcodecs
    >>> import numpy as np
    >>> x = np.array([[1, 3, 5], [[4, 3], [2, 1]], [[7, 9]]], dtype='object')
    >>> codec = numcodecs.VLenNDArray('<i4')
    >>> codec.decode(codec.encode(x))
    array([array([1, 3, 5], dtype=int32), array([[4, 3], [2, 1]], dtype=int32),
           array([[7, 9]]], dtype=int32)], dtype=object)

    See Also
    --------
    numcodecs.pickles.Pickle, numcodecs.json.JSON, numcodecs.msgpacks.MsgPack

    Notes
    -----
    The binary data for each array are packed into a parquet-style byte array.

    """

    codec_id = 'vlen-ndarray'

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
            object[:] shapes
            int[:] lengths
            int[:] ndims
            char* encv
            bytes b
            bytearray out
            char* data
            Buffer value_buffer
            object v

        # normalise input
        values = np.asarray(buf, dtype=object).reshape(-1, order='A')

        # determine number of items
        n_items = values.shape[0]

        # setup intermediates
        normed_values = np.empty(n_items, dtype=object)
        ndims = np.empty(n_items, dtype=np.intc)
        shapes = np.empty(n_items, dtype=object)
        lengths = np.empty(n_items, dtype=np.intc)

        # first iteration to convert to bytes
        data_length = 0
        for i in range(n_items):
            v = values[i]
            if v is None:
                v = np.array([], dtype=self.dtype)
                n = 0
                s = 0
            else:
                v = np.ascontiguousarray(v, self.dtype)
                n = v.ndim
                s = v.shape
                v = v.reshape(-1, order='A')
            l = v.nbytes
            normed_values[i] = v
            shapes[i] = s
            ndims[i] = n
            lengths[i] = l
            data_length += l + 4 * (n + 2)  # 4 bytes to store number of
                                            # dimensions, 4 bytes per
                                            # dimension to store dimension
                                            # and 4 bytes to store the length


        # setup output
        total_length = HEADER_LENGTH + data_length
        out = PyByteArray_FromStringAndSize(NULL, total_length)

        # write header
        data = PyByteArray_AS_STRING(out)
        store_le32(data, n_items)

        # second iteration, store data
        data += HEADER_LENGTH
        for i in range(n_items):
            l = lengths[i]
            s = shapes[i]
            n = ndims[i]
            store_le32(data, n)
            data += 4
            for j in range(n):
                store_le32(data, s[j])
                data += 4
            store_le32(data, l)
            data += 4
            value_buffer = Buffer(normed_values[i], PyBUF_ANY_CONTIGUOUS)
            encv = value_buffer.ptr
            memcpy(data, encv, l)
            data += l
            value_buffer.release()

        return out

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def decode(self, buf, out=None):
        cdef:
            Buffer input_buffer
            char* data
            char* data_end
            Py_ssize_t i, l, n_items, data_length, input_length

        # accept any buffer
        buf = ensure_contiguous_ndarray(buf)
        input_buffer = Buffer(buf, PyBUF_ANY_CONTIGUOUS)
        input_length = input_buffer.nbytes

        # sanity checks
        if input_length < HEADER_LENGTH:
            raise ValueError('corrupt buffer, missing or truncated header')

        # obtain input data pointer
        data = input_buffer.ptr
        data_end = data + input_length

        # load number of items
        n_items = load_le32(data)

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
            n = load_le32(data)
            data += 4
            s = np.empty(n, dtype=np.intc)
            for j in range(n):
                if data + 4 > data_end:
                    raise ValueError('corrupt buffer, data seem truncated')
                s[j] = load_le32(data)
                data += 4
            if data + 4 > data_end:
                raise ValueError('corrupt buffer, data seem truncated')
            l = load_le32(data)
            data += 4
            if data + l > data_end:
                raise ValueError('corrupt buffer, data seem truncated')
            d = np.frombuffer(data[:l], dtype=self.dtype)
            out[i] = d.reshape(s,  order='A')
            data += l

        return out
