# cython: embedsignature=True
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# cython: language_level=3
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release


from .compat import ensure_contiguous_ndarray


cdef class Buffer:
    """Convenience class for buffer interface."""

    def __cinit__(self, obj, flags):
        PyObject_GetBuffer(obj, &(self.buffer), flags)
        self.acquired = True
        self.ptr = <char *> self.buffer.buf
        self.itemsize = self.buffer.itemsize
        self.nbytes = self.buffer.len

    cpdef release(self):
        if self.acquired:
            PyBuffer_Release(&(self.buffer))
            self.acquired = False

    def __dealloc__(self):
        self.release()
