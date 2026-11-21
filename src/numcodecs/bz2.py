import bz2 as _bz2
from typing import cast, overload

from numcodecs.abc import Codec, CodecConfig
from numcodecs.compat import ensure_contiguous_ndarray, ndarray_copy
from numcodecs.ndarray_like import BufferLike


class BZ2Config(CodecConfig):
    """The configuration of a `BZ2` codec, as returned by `BZ2.get_config`."""

    level: int


class BZ2(Codec):
    """Codec providing compression using bzip2 via the Python standard library.

    Parameters
    ----------
    level : int
        Compression level.

    """

    codec_id = 'bz2'

    def __init__(self, level: int = 1) -> None:
        self.level = level

    def get_config(self) -> BZ2Config:
        return cast(BZ2Config, super().get_config())

    def encode(self, buf: BufferLike) -> bytes:
        # normalise input
        arr = ensure_contiguous_ndarray(buf)

        # do compression
        return _bz2.compress(arr, self.level)

    @overload
    def decode(self, buf: BufferLike, out: None = None) -> bytes: ...
    @overload
    def decode[OutT: BufferLike](self, buf: BufferLike, out: OutT) -> OutT: ...
    def decode(self, buf: BufferLike, out: BufferLike | None = None) -> BufferLike:
        # normalise inputs
        arr = ensure_contiguous_ndarray(buf)
        if out is not None:
            ensure_contiguous_ndarray(out)  # validate; decode into `out` itself below

        # N.B., bz2 cannot handle ndarray directly because of truth testing issues
        mem = memoryview(arr)

        # do decompression
        dec = _bz2.decompress(mem)

        # handle destination - Python standard library bz2 module does not
        # support direct decompression into buffer, so we have to copy into
        # out if given
        return ndarray_copy(dec, out)
