from cpython cimport array
import array


cdef class Buffer:
    cdef:
        char *ptr
        Py_buffer buffer
        size_t nbytes
        size_t itemsize
        array.array arr
        bint new_buffer
        bint released

    cpdef release(self)
