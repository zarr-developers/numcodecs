# -*- coding: utf-8 -*-
# cython: embedsignature=True
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
from __future__ import absolute_import, print_function, division
import struct


import cython
cimport cython
import numpy as np
cimport numpy as np
from .abc import Codec
from .compat_ext cimport Buffer
from .compat_ext import Buffer
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


# 8 bytes to store number of items
# 8 bytes to store data length
cdef Py_ssize_t HEADER_LENGTH = 16


def write_header(buf, n_items, data_length):
    struct.pack_into('<QQ', buf, 0, n_items, data_length)


def read_header(buf):
    return struct.unpack_from('<QQ', buf, 0)


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
            Py_ssize_t i, n_items, l, data_length, total_length
            np.ndarray[object] unicode_objects
            np.ndarray[object] encoded_values
            np.uint32_t[:] encoded_lengths
            char* encv
            bytes b
            bytearray out
            char* data
            object u

        # normalise input
        unicode_objects = np.asanyarray(buf, dtype=object).reshape(-1, order='A')

        # determine number of items
        n_items = unicode_objects.shape[0]

        # setup intermediates
        encoded_values = np.empty(n_items, dtype=object)
        encoded_lengths = np.empty(n_items, dtype='u4')

        # first iteration to convert to bytes
        data_length = 0
        for i in range(n_items):
            u = unicode_objects[i]
            if not PyUnicode_Check(u):
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
        write_header(out, n_items, data_length)

        # second iteration, store data
        data = PyByteArray_AS_STRING(out) + HEADER_LENGTH
        for i in range(n_items):
            l = encoded_lengths[i]
            store_le32(data, l)
            data += 4
            encv = PyBytes_AS_STRING(encoded_values[i])
            memcpy(data, encv, l)
            data += l

        return out

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def decode(self, buf, out=None):
        cdef:
            Buffer input_buffer
            char* data
            Py_ssize_t i, n_items, l, data_length, input_length
            np.ndarray[object, ndim=1] result

        # accept any buffer
        input_buffer = Buffer(buf, PyBUF_ANY_CONTIGUOUS)
        input_length = input_buffer.nbytes

        # sanity checks
        if input_length < HEADER_LENGTH:
            raise ValueError('corrupt buffer, missing or truncated header')

        # load number of items
        n_items, data_length = read_header(buf)

        # sanity check
        if input_length < data_length + HEADER_LENGTH:
            raise ValueError('corrupt buffer, data are truncated')

        # position input data pointer
        data = input_buffer.ptr + HEADER_LENGTH

        if out is not None:

            if not isinstance(out, np.ndarray):
                raise TypeError('out must be 1-dimensional array')
            if out.dtype != object:
                raise ValueError('out must be object array')
            out = out.reshape(-1, order='A')
            if out.shape[0] < n_items:
                raise ValueError('out is too small')

            # iterate and decode - N.B., use a separate loop and do not try to cast `out`
            # as np.ndarray[object] as this causes segfaults, possibly similar to
            # https://github.com/cython/cython/issues/1608
            for i in range(n_items):
                l = load_le32(data)
                data += 4
                out[i] = PyUnicode_FromStringAndSize(data, l)
                data += l

            return out

        else:

            # setup output
            result = np.empty(n_items, dtype=object)

            # iterate and decode - slightly faster as can use typed `result` variable
            for i in range(n_items):
                l = load_le32(data)
                data += 4
                result[i] = PyUnicode_FromStringAndSize(data, l)
                data += l

            return result


class VLenBytes(Codec):
    """Encode variable-length byte string objects.

    Examples
    --------
    >>> import numcodecs
    >>> import numpy as np
    >>> x = np.array([b'foo', b'bar', b'baz'], dtype='object')
    >>> codec = numcodecs.VLenUTF8()
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
            Py_ssize_t i, n_items, l, data_length, total_length
            np.ndarray[object] values
            np.uint32_t[:] lengths
            char* encv
            object b
            bytearray out
            char* data

        # normalise input
        values = np.asanyarray(buf, dtype=object).reshape(-1, order='A')

        # determine number of items
        n_items = values.shape[0]

        # setup intermediates
        lengths = np.empty(n_items, dtype='u4')

        # first iteration to find lengths
        data_length = 0
        for i in range(n_items):
            b = values[i]
            if not PyBytes_Check(b):
                raise TypeError('expected byte string, found %r' % b)
            l = PyBytes_GET_SIZE(b)
            data_length += l + 4  # 4 bytes to store item length
            lengths[i] = l

        # setup output
        total_length = HEADER_LENGTH + data_length
        out = PyByteArray_FromStringAndSize(NULL, total_length)

        # write header
        write_header(out, n_items, data_length)

        # second iteration, store data
        data = PyByteArray_AS_STRING(out) + HEADER_LENGTH
        for i in range(n_items):
            l = lengths[i]
            store_le32(data, l)
            data += 4
            encv = PyBytes_AS_STRING(values[i])
            memcpy(data, encv, l)
            data += l

        return out

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def decode(self, buf, out=None):
        cdef:
            Buffer input_buffer
            char* data
            Py_ssize_t i, n_items, l, data_length, input_length
            np.ndarray[object, ndim=1] result

        # accept any buffer
        input_buffer = Buffer(buf, PyBUF_ANY_CONTIGUOUS)
        input_length = input_buffer.nbytes

        # sanity checks
        if input_length < HEADER_LENGTH:
            raise ValueError('corrupt buffer, missing or truncated header')

        # load number of items
        n_items, data_length = read_header(buf)

        # sanity check
        if input_length < data_length + HEADER_LENGTH:
            raise ValueError('corrupt buffer, data are truncated')

        # position input data pointer
        data = input_buffer.ptr + HEADER_LENGTH

        if out is not None:

            if not isinstance(out, np.ndarray):
                raise TypeError('out must be 1-dimensional array')
            if out.dtype != object:
                raise ValueError('out must be object array')
            out = out.reshape(-1, order='A')
            if out.shape[0] < n_items:
                raise ValueError('out is too small')

            # iterate and decode - N.B., use a separate loop and do not try to cast `out`
            # as np.ndarray[object] as this causes segfaults, possibly similar to
            # https://github.com/cython/cython/issues/1608
            for i in range(n_items):
                l = load_le32(data)
                data += 4
                out[i] = PyBytes_FromStringAndSize(data, l)
                data += l

            return out

        else:

            # setup output
            result = np.empty(n_items, dtype=object)

            # iterate and decode - slightly faster as can use typed `result` variable
            for i in range(n_items):
                l = load_le32(data)
                data += 4
                result[i] = PyBytes_FromStringAndSize(data, l)
                data += l

            return result
