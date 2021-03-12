# cython: embedsignature=True
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# cython: language_level=3

import numpy as np
from .compat import ensure_contiguous_ndarray
from .abc import Codec

cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _doShuffle(const unsigned char[::1] src, unsigned char[::1] des, int element_size) nogil:
    cdef Py_ssize_t count, i, j, offset, byte_index
    count = len(src) // element_size
    for i in range(count):
        offset = i*element_size
        for byte_index in range(element_size):
            j = byte_index*count + i
            des[j] = src[offset + byte_index]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _doUnshuffle(const unsigned char[::1] src, unsigned char[::1] des, int element_size) nogil:
    cdef Py_ssize_t count, i, j, offset, byte_index
    count = len(src) // element_size
    for i in range(element_size):
        offset = i*count
        for byte_index in range(count):
            j = byte_index*element_size + i
            des[j] = src[offset+byte_index]


class Shuffle(Codec):
    """Codec providing shuffle

    Parameters
    ----------
    elementsize : int
        Size in bytes of the array elements.  Default = 4

    """

    codec_id = 'shuffle'

    def __init__(self, elementsize=4):
        self.elementsize = elementsize

    def encode(self, buf, out=None):
        buf = ensure_contiguous_ndarray(buf)

        if out is None:
            out = np.zeros(buf.nbytes, dtype='uint8')
        else:
            out = ensure_contiguous_ndarray(out)

        if self.elementsize <= 1:
            out.view(buf.dtype)[:len(buf)] = buf[:]
            return out  # no shuffling needed

        if buf.nbytes % self.elementsize != 0:
            raise ValueError("Shuffle buffer is not an integer multiple of elementsize")

        _doShuffle(buf.view("uint8"), out.view("uint8"), self.elementsize)

        return out

    def decode(self, buf, out=None):
        buf = ensure_contiguous_ndarray(buf)

        if out is None:
            out = np.zeros(buf.nbytes, dtype='uint8')
        else:
            out = ensure_contiguous_ndarray(out)

        if self.elementsize <= 1:
            out.view(buf.dtype)[:len(buf)] = buf[:]
            return out  # no shuffling needed

        if buf.nbytes % self.elementsize != 0:
            raise ValueError("Shuffle buffer is not an integer multiple of elementsize")

        _doUnshuffle(buf.view("uint8"), out.view("uint8"), self.elementsize)

        return out

    def __repr__(self):
        r = '%s(elementsize=%s)' % \
            (type(self).__name__,
             self.elementsize)
        return r
