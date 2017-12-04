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
from cpython cimport (PyUnicode_AsUTF8String, PyBytes_AS_STRING, PyBytes_GET_SIZE,
                      PyUnicode_DecodeUTF8)
from libc.string cimport memcpy
cdef extern from "Python.h":
    bytearray PyByteArray_FromStringAndSize(char *v, Py_ssize_t l)
    char* PyByteArray_AS_STRING(object string)


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
            np.ndarray[object] bytes_objects
            np.uint32_t[:] bytes_lengths
            bytes b
            bytearray out
            char* data

        # TODO value checks
        unicode_objects = buf

        # determine number of items
        n_items = unicode_objects.shape[0]

        # setup intermediates
        bytes_objects = np.empty(n_items, dtype=object)
        bytes_lengths = np.empty(n_items, dtype='u4')

        # first iteration to convert to bytes
        data_length = 0
        for i in range(n_items):
            b = PyUnicode_AsUTF8String(unicode_objects[i])
            bytes_objects[i] = b
            l = PyBytes_GET_SIZE(b)
            data_length += l + 4  # 4 bytes to store item length
            bytes_lengths[i] = l

        # setup output
        total_length = HEADER_LENGTH + data_length
        out = PyByteArray_FromStringAndSize(NULL, total_length)

        # write header
        write_header(out, n_items, data_length)

        # second interation, store data
        data = PyByteArray_AS_STRING(out) + HEADER_LENGTH
        for i in range(n_items):
            l = bytes_lengths[i]
            store_le32(data, l)
            data += 4
            memcpy(data, PyBytes_AS_STRING(bytes_objects[i]), l)
            data += l

        return out

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def decode(self, buf, out=None):
        cdef:
            Buffer input_buffer
            char* data
            Py_ssize_t i, n_items, l, data_length, input_length
            np.ndarray[object] result
            object b

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

        # setup output
        if out is not None:
            # TODO value checks
            result = out
        else:
            result = np.empty(n_items, dtype=object)

        # iterate over items, decoding
        data = input_buffer.ptr + HEADER_LENGTH
        for i in range(n_items):
            l = load_le32(data)
            data += 4
            b = PyUnicode_DecodeUTF8(data, l, NULL)
            result[i] = b
            data += l

        return result
