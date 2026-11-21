import base64 as _base64
from typing import cast, overload

from .abc import Codec, CodecConfig
from .compat import ensure_contiguous_ndarray, ndarray_copy
from .ndarray_like import BufferLike


class Base64(Codec):
    """Codec providing base64 compression via the Python standard library."""

    codec_id = "base64"

    def get_config(self) -> CodecConfig:
        return cast(CodecConfig, super().get_config())

    def encode(self, buf: BufferLike) -> bytes:
        # normalise inputs
        arr = ensure_contiguous_ndarray(buf)
        # do compression
        return _base64.standard_b64encode(arr)

    @overload
    def decode(self, buf: BufferLike, out: None = None) -> bytes: ...
    @overload
    def decode[OutT: BufferLike](self, buf: BufferLike, out: OutT) -> OutT: ...
    def decode(self, buf: BufferLike, out: BufferLike | None = None) -> BufferLike:
        # normalise inputs
        arr = ensure_contiguous_ndarray(buf)
        if out is not None:
            ensure_contiguous_ndarray(out)  # validate; decode into `out` itself below
        # do decompression
        decompressed = _base64.standard_b64decode(arr)
        # handle destination
        return ndarray_copy(decompressed, out)
