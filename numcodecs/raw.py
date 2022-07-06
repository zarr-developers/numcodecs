from .abc import Codec
from .compat import ensure_bytes, ensure_contiguous_ndarray, ndarray_copy


class Raw(Codec):
    codec_id: str = "raw"

    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf)
        return buf.tobytes()

    def decode(self, buf: bytes, out=None):
        buf = ensure_bytes(buf)
        if out is None:
            return buf
        else:
            out_view = ensure_contiguous_ndarray(out)
            out = ndarray_copy(buf, out_view)
            return out
