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
from cpython.buffer cimport PyBUF_ANY_CONTIGUOUS
from cpython cimport PyObject
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
cdef extern from "Python.h":
    bytearray PyByteArray_FromStringAndSize(char *v, Py_ssize_t l)
    char* PyByteArray_AS_STRING(object string)
    char* PyUnicode_AsUTF8AndSize(object text, Py_ssize_t *size)
    PyObject* PyUnicode_FromStringAndSize(const char *u, Py_ssize_t size)
    PyObject* PyList_New(Py_ssize_t l)
    void PyList_SET_ITEM(PyObject *l, Py_ssize_t i, PyObject *o)

from .abc import Codec
from .compat_ext cimport Buffer
from .compat_ext import Buffer
cdef extern from "stdint_compat.h":
    void store_le32(char *c, int y)
    int load_le32(const char *c)
    void store_le64(char *c, int y)
    int load_le64(const char *c)


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
            char** encoded_values
            int* encoded_lengths
            bytes b
            bytearray out
            char* data

        # normalise input
        buf = np.asanyarray(buf, dtype=object).reshape(-1, order='A')
        unicode_objects = buf

        # determine number of items
        n_items = unicode_objects.shape[0]

        # setup intermediates
        encoded_values = <char**> malloc(n_items * sizeof(char*))
        encoded_lengths = <int*> malloc(n_items * sizeof(int))

        try:

            # first iteration to convert to bytes
            data_length = 0
            for i in range(n_items):
                encoded_values[i] = PyUnicode_AsUTF8AndSize(unicode_objects[i], &l)
                data_length += l + 4  # 4 bytes to store item length
                encoded_lengths[i] = l

            # setup output
            total_length = HEADER_LENGTH + data_length
            out = PyByteArray_FromStringAndSize(NULL, total_length)

            # write header
            write_header(out, n_items, data_length)

            # second interation, store data
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

    @cython.wraparound(True)
    @cython.boundscheck(True)
    def decode(self, buf, out=None):
        cdef:
            Buffer input_buffer
            char* data
            Py_ssize_t i, n_items, l, data_length, input_length
            # np.ndarray[object, ndim=1] result
            PyObject* result
            PyObject* b
            # object b

        # accept any buffer
        input_buffer = Buffer(buf, PyBUF_ANY_CONTIGUOUS)
        input_length = input_buffer.nbytes

        # sanity checks
        if input_length < HEADER_LENGTH:
            raise RuntimeError('corrupt buffer, missing or truncated header')

        # load number of items
        n_items, data_length = read_header(buf)

        # sanity check
        if input_length < data_length + HEADER_LENGTH:
            raise RuntimeError('corrupt buffer, data are truncated')

        # position input data pointer
        data = input_buffer.ptr + HEADER_LENGTH

        if out is not None:
            raise NotImplementedError

            # if not isinstance(out, np.ndarray):
            #     raise ValueError('out must be 1-dimensional array')
            # if out.ndim != 1:
            #     raise ValueError('out must be 1-dimensional array')
            # if out.dtype != object:
            #     raise ValueError('out must be object array')
            # if out.shape[0] < n_items:
            #     raise ValueError('out is too small')
            #
            # # iterate and decode
            # for i in range(n_items):
            #     l = load_le32(data)
            #     data += 4
            #     b = PyUnicode_FromStringAndSize(data, l)
            #     out[i] = b
            #     data += l
            #
            # return out

        else:

            # setup output
            # result = np.empty(n_items, dtype=object)
            result = PyList_New(n_items)

            # iterate and decode
            for i in range(n_items):
                l = load_le32(data)
                data += 4
                b = PyUnicode_FromStringAndSize(data, l)
                # result[i] = b
                PyList_SET_ITEM(result, i, b)
                data += l

            return np.asarray(<list> result, dtype=object)
