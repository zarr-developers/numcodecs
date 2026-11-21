import zlib as _zlib
from typing import cast, overload

from .abc import Codec, CodecConfig
from .compat import ensure_contiguous_ndarray, ndarray_copy
from .ndarray_like import BufferLike


class ZlibConfig(CodecConfig):
    """The configuration of a `Zlib` codec, as returned by `Zlib.get_config`."""

    level: int


class Zlib(Codec):
    """Codec providing compression using zlib via the Python standard library.

    Parameters
    ----------
    level : int
        Compression level.

    """

    codec_id = 'zlib'

    def __init__(self, level: int = 1) -> None:
        self.level = level

    def get_config(self) -> ZlibConfig:
        return cast(ZlibConfig, super().get_config())

    def encode(self, buf: BufferLike) -> bytes:
        # normalise inputs
        arr = ensure_contiguous_ndarray(buf)

        # do compression
        return _zlib.compress(arr, self.level)

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
        dec = _zlib.decompress(arr)

        # handle destination - Python standard library zlib module does not
        # support direct decompression into buffer, so we have to copy into
        # out if given
        return ndarray_copy(dec, out)
