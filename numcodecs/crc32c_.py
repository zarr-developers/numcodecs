import numpy as np
from crc32c import crc32c

from .abc import Codec
from .compat import ensure_bytes, ensure_contiguous_ndarray


class Crc32c(Codec):
    """Codec that adds a CRC32C checksum to the encoded data."""

    codec_id = 'crc32c'

    def encode(self, buf):
        # normalise inputs
        buf = ensure_contiguous_ndarray(buf)
        buf = ensure_bytes(buf)
        print(buf)
        checksum = crc32c(buf)
        print(checksum)

        checksum_arr = np.zeros(1, dtype="<u4")
        checksum_arr[0] = checksum
        return buf + checksum_arr.tobytes()

    def decode(self, buf, out=None):
        # normalise inputs
        buf = ensure_bytes(buf)
        computed_checksum = crc32c(memoryview(buf)[:-4])

        if len(buf) < 4:
            raise ValueError("Input buffer is too short to contain a CRC32C checksum.")

        if out is not None:
            out_view = ensure_contiguous_ndarray(out).view("b")
            if len(out_view) < len(buf) - 4:
                raise ValueError("Output buffer is too small to contain decoded data.")
            elif len(out_view) > len(buf) - 4:
                raise ValueError("Output buffer is too large to contain decoded data.")
            out_view[:] = np.frombuffer(buf, "b", count=(len(buf) - 4), offset=0)
        else:
            out = buf[:-4]

        stored_checksum = np.frombuffer(buf, "<u4", offset=(len(buf) - 4))[0]
        if computed_checksum != stored_checksum:
            raise ValueError(
                f"Stored and computed checksum do not match. Stored: {stored_checksum}. Computed: {computed_checksum}."
            )

        return out
