# cython: language_level=3


cdef extern from *:
    """
    #define PyBytes_RESIZE(b, n) _PyBytes_Resize(&b, n)
    """
    int PyBytes_RESIZE(object b, Py_ssize_t n) except -1


cpdef memoryview ensure_continguous_memoryview(obj)
