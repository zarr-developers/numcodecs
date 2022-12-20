import struct

from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray

from libc.stdint cimport uint8_t, uint16_t, uint32_t

cdef extern from "_fletcher.c":
    uint32_t H5_checksum_fletcher32(const void *_data, size_t _len)


class Fletcher32(Codec):
    """The fletcher checksum with 16-bit words and 32-bit output

    With this codec, the checksum is concatenated on the end of the data
    bytes when encoded. At decode time, the checksum is performed on
    the data portion and compared with the four-byte checksum, raising
    ValueError if inconsistent.
    """

    codec_id = "fletcher32"

    def encode(self, buf):
        """Return buffer plus 4-byte fletcher checksum"""
        buf = ensure_contiguous_ndarray(buf).ravel().view('uint8')
        cdef const uint8_t[::1] b_ptr = buf
        val = H5_checksum_fletcher32(&b_ptr[0], buf.nbytes)
        return buf.tobytes() + struct.pack("<I", val)

    def decode(self, buf, out=None):
        """Check fletcher checksum, and return buffer without it"""
        b = ensure_contiguous_ndarray(buf).view('uint8')
        cdef const uint8_t[::1] b_ptr = b
        val = H5_checksum_fletcher32(&b_ptr[0], b.nbytes - 4)
        found = b[-4:].view("<u4")[0]
        if val != found:
            raise RuntimeError(
                f"The fletcher32 checksum of the data ({val}) did not"
                f" match the expected checksum ({found}).\n"
                "This could be a sign that the data has been corrupted."
            )
        if out:
            out.view("uint8")[:] = b[:-4]
            return out
        return memoryview(b[:-4])
