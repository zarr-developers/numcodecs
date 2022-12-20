# cython: boundscheck=False
# cython: wraparound=False
# cython: overflowcheck=False
# cython: cdivision=True

import struct
import numpy as np

from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray

from libc.stdint cimport uint8_t, uint16_t, uint32_t

cpdef uint32_t fletcher32(const uint16_t[::1] data):
    cdef:
        uint32_t sum1 = 0
        uint32_t sum2 = 0
        int index
        int size = data.shape[0]

    for index in range(0, size):
        sum1 = (sum1 + data[index]) % 0xffff
        sum2 = (sum2 + sum1) % 0xffff

    return (sum2 << 16) | sum1


class Fletcher32(Codec):
    codec_id = "fletcher32"

    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf).ravel()
        if len(buf) % 2:
            # rare, odd size of bytes data only
            arr = np.frombuffer(buf.tobytes() + b"\x00", dtype="uint16")
            val = fletcher32(arr)
        else:
            val = fletcher32(buf.view('uint16'))
        return buf.tobytes() + struct.pack("<I", val)

    def decode(self, buf, out=None):
        b = ensure_contiguous_ndarray(buf).view('uint8')
        if len(buf) % 2:
            # rare, odd size of bytes data only
            arr = np.frombuffer(b.tobytes() + b"\x00", dtype="uint16")
            val = fletcher32(arr)
        else:
            val = fletcher32(b[:-4].view('uint16'))
        found = b[-4:].view('uint32')[0]
        if val != found:
            raise ValueError(
                f"The flecher32 checksum of the data ({found}) did not match the expected checksum ({val}). "
                "This could be a sign that the data has been corrupted."
            )
        if out:
            out.view("uint8")[:] = b[:-4]
            return out
        return memoryview(b[:-4])
