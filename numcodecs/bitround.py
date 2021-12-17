import numpy as np


from .abc import Codec
from .compat import ensure_ndarray, ndarray_copy


class BitRound(Codec):
    codec_id = 'bitround'

    def __init__(self, keepbits: int):
        self.keepbits = keepbits

    def encode(self, buf):
        # TODO: figure out if we need to make a copy
        # Currently this appears to be overwriting the input buffer
        # Is that the right behavior?
        a = ensure_ndarray(buf).view()
        assert a.dtype == np.float32
        b = a.view(dtype=np.int32)
        maskbits = 23 - self.keepbits
        mask = (0xFFFFFFFF >> maskbits) << maskbits
        half_quantum1 = (1 << (maskbits - 1)) - 1
        b += ((b >> maskbits) & 1) + half_quantum1
        b &= mask
        return b

    def decode(self, buf, out=None):
        data = ensure_ndarray(buf).view(np.float32)
        out = ndarray_copy(data, out)
        return out
