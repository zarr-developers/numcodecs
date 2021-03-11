import numpy as np
from .compat import ensure_bytes, ensure_contiguous_ndarray, ndarray_copy, ensure_ndarray
from .abc import Codec


cdef _doShuffle(const unsigned char[:] src, unsigned char[:] des, int element_size):
    cdef int count, i, j, offset, byte_index
    count = len(src) // element_size
    for i in range(count):
        offset = i*element_size
        e = src[offset:(offset+element_size)]
        for byte_index in range(element_size):
            j = byte_index*count + i
            des[j] = e[byte_index]
    return des


cdef _doUnshuffle(const unsigned char[:] src, unsigned char[:] des, int element_size):
    cdef int count, i, j, offset, byte_index
    count = len(src) // element_size
    for i in range(element_size):
        offset = i*count
        e = src[offset:(offset+count)]
        for byte_index in range(count):
            j = byte_index*element_size + i
            des[j] = e[byte_index]
    return des


def _shuffle(element_size, buf, out):
    if element_size <= 1:
        out.view(buf.dtype)[:len(buf)] = buf[:]
        return out  # no shuffling needed

    buf_size = buf.nbytes
    if buf_size % element_size != 0:
        raise ValueError("Shuffle buffer is not an integer multiple of elementsize")

    _doShuffle(buf.view("uint8"), out.view("uint8"), element_size)

    return out


def _unshuffle(element_size, buf, out):
    if element_size <= 1:
        out.view(buf.dtype)[:len(buf)] = buf[:]
        return out  # no shuffling needed

    buf_size = buf.nbytes
    if buf_size % element_size != 0:
        raise ValueError("Shuffle buffer is not an integer multiple of elementsize")

    _doUnshuffle(buf.view("uint8"), out.view("uint8"), element_size)

    return out


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
        return _shuffle(self.elementsize, buf, out)

    def decode(self, buf, out=None):
        buf = ensure_contiguous_ndarray(buf)
        if out is None:
            out = np.zeros(buf.nbytes, dtype='uint8')
        else:
            out = ensure_contiguous_ndarray(out)
        return _unshuffle(self.elementsize, buf, out)

    def __repr__(self):
        r = '%s(elementsize=%s)' % \
            (type(self).__name__,
             self.elementsize)
        return r
