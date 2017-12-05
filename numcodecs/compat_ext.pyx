# -*- coding: utf-8 -*-
# cython: embedsignature=True
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
from __future__ import absolute_import, print_function, division
import sys
from cpython cimport array
import array
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release


PY2 = sys.version_info[0] == 2


cdef class Buffer:
    """Compatibility class to work around fact that array.array does not support
    new-style buffer interface in PY2."""

    def __cinit__(self, obj, flags):
        self.released = False
        if hasattr(obj, 'dtype'):
            if obj.dtype.kind in 'Mm':
                obj = obj.view('u8')
            elif obj.dtype.kind == 'O':
                raise ValueError('cannot obtain buffer from object array')
        if PY2 and isinstance(obj, array.array):
            self.new_buffer = False
            self.arr = obj
            self.ptr = <char *> self.arr.data.as_voidptr
            self.itemsize = self.arr.itemsize
            self.nbytes = self.arr.buffer_info()[1] * self.itemsize
        else:
            self.new_buffer = True
            PyObject_GetBuffer(obj, &(self.buffer), flags)
            self.ptr = <char *> self.buffer.buf
            self.itemsize = self.buffer.itemsize
            self.nbytes = self.buffer.len

    cpdef release(self):
        if self.new_buffer and not self.released:
            PyBuffer_Release(&(self.buffer))
            self.released = True

    def __dealloc__(self):
        self.release()
