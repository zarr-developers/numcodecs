# -*- coding: utf-8 -*-
# cython: embedsignature=True
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
from __future__ import absolute_import, print_function, division
import struct


import cython
import numpy as np
cimport numpy as np
from .abc import Codec
from .compat_ext cimport Buffer
from .compat_ext import Buffer
from cpython cimport PyBytes_GET_SIZE, PyBytes_AS_STRING
from cpython.buffer cimport PyBUF_ANY_CONTIGUOUS
from cpython cimport PyUnicode_AsUTF8String
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy


cdef extern from "stdint_compat.h":
    void store_le32(char *c, int y)
    int load_le32(const char *c)


cdef extern from "Python.h":
    bytearray PyByteArray_FromStringAndSize(char *v, Py_ssize_t l)
    char* PyByteArray_AS_STRING(object string)
    object PyUnicode_FromStringAndSize(const char *u, Py_ssize_t size)
    int PyUnicode_Check(object text)


# noinspection PyUnresolvedReferences
IF COMPILE_PY2:
    cdef char* PyUnicode_AsUTF8AndSize(object u, Py_ssize_t* l):
        cdef:
            bytes b
            char* encv
        b = PyUnicode_AsUTF8String(u)
        encv = PyBytes_AS_STRING(b)
        l[0] = PyBytes_GET_SIZE(b)
        return encv


# noinspection PyUnresolvedReferences
IF COMPILE_PY3:
    cdef extern from "Python.h":
        char* PyUnicode_AsUTF8AndSize(object text, Py_ssize_t *size)


# 8 bytes to store number of items
# 8 bytes to store data length
cdef Py_ssize_t HEADER_LENGTH = 16


def write_header(buf, n_items, data_length):
    struct.pack_into('<QQ', buf, 0, n_items, data_length)


def read_header(buf):
    return struct.unpack_from('<QQ', buf, 0)


class VLenUTF8(Codec):

    codec_id = 'vlen-utf8'

    def __init__(self):
        pass

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def encode(self, buf):
        cdef:
            Py_ssize_t i, n_items, l, data_length, total_length
            np.ndarray[object] unicode_objects
            char* encv
            char** encoded_values
            int* encoded_lengths
            bytes b
            bytearray out
            char* data
            object u

        # normalise input
        unicode_objects = np.asanyarray(buf, dtype=object).reshape(-1, order='A')

        # determine number of items
        n_items = unicode_objects.shape[0]

        # setup intermediates
        encoded_values = <char**> malloc(n_items * sizeof(char*))
        encoded_lengths = <int*> malloc(n_items * sizeof(int))

        try:

            # first iteration to convert to bytes
            data_length = 0
            for i in range(n_items):
                u = unicode_objects[i]
                if not PyUnicode_Check(u):
                    raise TypeError('expected unicode string, found %r' % u)
                # IF COMPILE_PY2:
                #     b = PyUnicode_AsUTF8String(u)
                #     encv = PyBytes_AS_STRING(b)
                #     l = PyBytes_GET_SIZE(b)
                # IF COMPILE_PY3:
                encv = PyUnicode_AsUTF8AndSize(u, &l)
                encoded_values[i] = encv
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
                memcpy(data, encoded_values[i], l)
                data += l

        finally:
            free(encoded_values)
            free(encoded_lengths)

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

            # iterate and decode - N.B., use a separate loop and do not try to cast out
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

            # iterate and decode - slightly faster as can use typed result variable
            for i in range(n_items):
                l = load_le32(data)
                data += 4
                result[i] = PyUnicode_FromStringAndSize(data, l)
                data += l

            return result
