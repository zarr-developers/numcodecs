
# The _doShuffle and _doUnshuffle functions have been lifted from HSDS which 
# includes the following notice:
##############################################################################
# Copyright by The HDF Group.                                                #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HSDS (HDF5 Scalable Data Service), Libraries and      #
# Utilities.  The full HSDS copyright notice, including                      #
# terms governing use, modification, and redistribution, is contained in     #
# the file COPYING, which can be found at the root of the source code        #
# distribution tree.  If you do not have access to this file, you may        #
# request a copy from help@hdfgroup.org.                                     #
##############################################################################

from numba import jit
import numpy as np
from .compat import ensure_bytes, ensure_contiguous_ndarray, ndarray_copy, ensure_ndarray
from .abc import Codec

@jit(nopython=True)
def _doShuffle(src, des, element_size):
    count = len(src) // element_size
    for i in range(count):
        offset = i*element_size
        e = src[offset:(offset+element_size)]
        for byte_index in range(element_size):
            j = byte_index*count + i
            des[j] = e[byte_index]
    return des

@jit(nopython=True)
def _doUnshuffle(src, des, element_size):
    count = len(src) // element_size
    for i in range(element_size):
        offset = i*count
        e = src[offset:(offset+count)]
        for byte_index in range(count):
            j = byte_index*element_size + i
            des[j] = e[byte_index]
    return des

def _shuffle(element_size, buf):
    if element_size <= 1:
        return  buf # no shuffling needed

    buf_size = buf.nbytes
    if buf_size % element_size != 0:
        raise ValueError("Shuffle buffer is not an integer multiple of elementsize")

    arr = np.zeros((buf_size,), dtype='u1')
    _doShuffle(ensure_bytes(buf), arr, element_size)

    return ensure_ndarray(arr)

def _unshuffle(element_size, buf):
    if element_size <= 1:
        return  buf # no shuffling needed

    buf_size = buf.nbytes
    if buf_size % element_size != 0:
        raise ValueError("Shuffle buffer is not an integer multiple of elementsize")

    arr = np.zeros((buf_size,), dtype='u1')
    _doUnshuffle(ensure_bytes(buf), arr, element_size)

    return ensure_ndarray(arr)


class Shuffle(Codec):
    """Codec providing shuffle using numba compiled shuffle functions.

    Parameters
    ----------
    elementsize : int
        Size in bytes of the array elements.  Default = 4

    See Also
    --------

    """

    codec_id = 'shuffle'

    def __init__(self, elementsize=4):
        self.elementsize = elementsize

    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf)
        return _shuffle(self.elementsize, buf)

    def decode(self, buf, out=None):
        buf = ensure_contiguous_ndarray(buf)
        dec = _unshuffle(self.elementsize, buf)

        # handle destination - couldnt work out how to directly write to
        # numpy array data from numba function, so we have to copy into
        # out if given
        return ndarray_copy(dec, out)

    def __repr__(self):
        r = '%s(elementsize=%s)' % \
            (type(self).__name__,
             self.elementsize)
        return r