# cython: language_level=3
# cython: overflowcheck=False
# cython: cdivision=True
import struct

from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray

from libc.stdint cimport uint8_t, uint16_t, uint32_t


cdef uint32_t _fletcher32(const uint8_t[::1] _data):
    # converted from
    # https://github.com/Unidata/netcdf-c/blob/main/plugins/H5checksum.c#L109
    cdef:
        const uint8_t *data = &_data[0]
        size_t _len = _data.shape[0]
        size_t len = _len / 2
        size_t tlen
        uint32_t sum1 = 0, sum2 = 0;


    while len:
        tlen = 360 if len > 360 else len
        len -= tlen
        while True:
            sum1 += <uint32_t>((<uint16_t>data[0]) << 8) | (<uint16_t>data[1])
            data += 2
            sum2 += sum1
            tlen -= 1
            if tlen < 1:
                break
        sum1 = (sum1 & 0xffff) + (sum1 >> 16)
        sum2 = (sum2 & 0xffff) + (sum2 >> 16)

    if _len % 2:
        sum1 += <uint32_t>((<uint16_t>(data[0])) << 8)
        sum2 += sum1
        sum1 = (sum1 & 0xffff) + (sum1 >> 16)
        sum2 = (sum2 & 0xffff) + (sum2 >> 16)

    sum1 = (sum1 & 0xffff) + (sum1 >> 16)
    sum2 = (sum2 & 0xffff) + (sum2 >> 16)

    return (sum2 << 16) | sum1


class Fletcher32(Codec):
    """The fletcher checksum with 16-bit words and 32-bit output

    This is the netCDF4/HED5 implementation, which is not equivalent
    to the one in wikipedia
    https://github.com/Unidata/netcdf-c/blob/main/plugins/H5checksum.c#L95

    With this codec, the checksum is concatenated on the end of the data
    bytes when encoded. At decode time, the checksum is performed on
    the data portion and compared with the four-byte checksum, raising
    RuntimeError if inconsistent.
    """

    codec_id = "fletcher32"

    def encode(self, buf):
        """Return buffer plus 4-byte fletcher checksum"""
        buf = ensure_contiguous_ndarray(buf).ravel().view('uint8')
        cdef const uint8_t[::1] b_ptr = buf
        val = _fletcher32(b_ptr)
        return buf.tobytes() + struct.pack("<I", val)

    def decode(self, buf, out=None):
        """Check fletcher checksum, and return buffer without it"""
        b = ensure_contiguous_ndarray(buf).view('uint8')
        cdef const uint8_t[::1] b_ptr = b[:-4]
        val = _fletcher32(b_ptr)
        found = b[-4:].view("<u4")[0]
        if val != found:
            raise RuntimeError(
                f"The fletcher32 checksum of the data ({val}) did not"
                f" match the expected checksum ({found}).\n"
                "This could be a sign that the data has been corrupted."
            )
        if out is not None:
            out.view("uint8")[:] = b[:-4]
            return out
        return memoryview(b[:-4])
