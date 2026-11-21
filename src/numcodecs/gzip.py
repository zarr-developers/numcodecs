import gzip as _gzip
import io
from typing import cast, overload

from .abc import Codec, CodecConfig
from .compat import ensure_bytes, ensure_contiguous_ndarray
from .ndarray_like import BufferLike


class GZipConfig(CodecConfig):
    """The configuration of a `GZip` codec, as returned by `GZip.get_config`."""

    level: int


class GZip(Codec):
    """Codec providing gzip compression using zlib via the Python standard library.

    Parameters
    ----------
    level : int
        Compression level.

    """

    codec_id = 'gzip'

    def __init__(self, level: int = 1) -> None:
        self.level = level

    def get_config(self) -> GZipConfig:
        return cast(GZipConfig, super().get_config())

    def encode(self, buf: BufferLike) -> bytes:
        # normalise inputs
        arr = ensure_contiguous_ndarray(buf)

        # do compression
        compressed = io.BytesIO()
        with _gzip.GzipFile(fileobj=compressed, mode='wb', compresslevel=self.level) as compressor:
            compressor.write(arr)
        return compressed.getvalue()

    @overload
    def decode(self, buf: BufferLike, out: None = None) -> bytes: ...
    @overload
    def decode[OutT: BufferLike](self, buf: BufferLike, out: OutT) -> OutT: ...
    def decode(self, buf: BufferLike, out: BufferLike | None = None) -> BufferLike:
        # normalise inputs
        # BytesIO only copies if the data is not of `bytes` type.
        # This allows `bytes` objects to pass through without copying.
        compressed = io.BytesIO(ensure_bytes(buf))

        # do decompression
        with _gzip.GzipFile(fileobj=compressed, mode='rb') as decompressor:
            if out is None:
                return decompressor.read()
            out_view = ensure_contiguous_ndarray(out)
            decompressor.readinto(out_view)
            if decompressor.read(1) != b'':
                raise ValueError("Unable to fit data into `out`")

        return out
