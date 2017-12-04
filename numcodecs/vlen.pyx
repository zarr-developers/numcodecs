# -*- coding: utf-8 -*-
# cython: embedsignature=True
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
from __future__ import absolute_import, print_function, division


import cython
import numpy as np
cimport numpy as np
from cpython.buffer cimport PyBUF_ANY_CONTIGUOUS
from cpython cimport (PyUnicode_AsUTF8String, PyBytes_GET_SIZE, PyBytes_AS_STRING,
                      PyBytes_FromStringAndSize, PyUnicode_DecodeUTF8)
from libc.string cimport memcpy
from .abc import Codec
from .compat_ext cimport Buffer
from .compat_ext import Buffer


cdef extern from "stdint_compat.h":
    cdef enum:
        UINT32_SIZE,
    void store_le32(char *c, int y)
    int load_le32(const char *c)


@cython.wraparound(False)
@cython.boundscheck(False)
class VLenUTF8(Codec):

    codec_id = 'vlen-utf8'

    def __init__(self):
        pass

    def encode(self, buf):
        cdef:
            Py_ssize_t i, n, data_length, header_length, total_length
            np.ndarray[object] unicode_objects
            np.ndarray[object] bytes_objects
            np.int32_t[:] bytes_lengths
            bytes b
            np.int32_t l
            bytes out
            char* data

        # TODO value checks
        unicode_objects = buf

        # determine number of items
        n = unicode_objects.shape[0]

        # setup intermediates
        bytes_objects = np.empty(n, dtype=object)
        bytes_lengths = np.empty(n, dtype='<i4')

        # 4 bytes to store number of items
        # 4 bytes to store total length - header
        header_length = 8
        data_length = 0

        # first iteration to convert to bytes
        for i in range(n):
            b = PyUnicode_AsUTF8String(unicode_objects[i])
            bytes_objects[i] = b
            l = PyBytes_GET_SIZE(b)
            data_length += l + 4  # 4 bytes to store item length
            bytes_lengths[i] = l

        # setup output
        total_length = header_length + data_length
        out = PyBytes_FromStringAndSize(NULL, total_length)
        data = PyBytes_AS_STRING(out)

        # store number of items
        store_le32(data, n)
        data += 4

        # store data length
        # TODO need larger size for this?
        store_le32(data, data_length)
        data += 4

        # second interation, store data
        for i in range(n):
            l = bytes_lengths[i]
            store_le32(data, l)
            data += 4
            memcpy(data, PyBytes_AS_STRING(bytes_objects[i]), l)
            data += l

        return out

    def decode(self, buf, out=None):
        cdef:
            Buffer input_buffer
            char* data
            Py_ssize_t i, n, data_length, input_length
            np.ndarray[object] result
            np.int32_t l

        # accept any buffer
        input_buffer = Buffer(buf, PyBUF_ANY_CONTIGUOUS)
        data = input_buffer.ptr
        input_length = input_buffer.nbytes

        # sanity checks
        if input_length < 8:
            raise RuntimeError('corrupt buffer, missing header')

        # load number of items
        n = load_le32(data)
        data += 4

        # load data length
        data_length = load_le32(data)
        data += 4

        # sanity check
        if input_length < data_length + 8:
            raise RuntimeError('corrupt buffer, data are truncated')

        # setup output
        if out is not None:
            # TODO value checks
            result = out
        else:
            result = np.empty(n, dtype=object)

        # iterate
        for i in range(n):
            l = load_le32(data)
            data += 4
            result[i] = PyUnicode_DecodeUTF8(data, l, NULL)
            data += l

        return result
