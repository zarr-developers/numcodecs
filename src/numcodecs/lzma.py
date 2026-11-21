from types import ModuleType
from typing import Any, cast, overload

_lzma: ModuleType | None = None
try:
    import lzma as _lzma
except ImportError:  # pragma: no cover
    try:  # noqa: SIM105
        from backports import lzma as _lzma  # type: ignore[no-redef]
    except ImportError:
        pass


if _lzma:
    from .abc import Codec, CodecConfig
    from .compat import ensure_contiguous_ndarray, ndarray_copy
    from .ndarray_like import BufferLike

    _lzma_module: ModuleType = _lzma  # narrowed from `ModuleType | None` for the methods below

    class LZMAConfig(CodecConfig):
        """The configuration of an `LZMA` codec, as returned by `LZMA.get_config`."""

        format: int
        check: int
        preset: int | None
        filters: list[dict[str, Any]] | None

    # noinspection PyShadowingBuiltins
    class LZMA(Codec):
        """Codec providing compression using lzma via the Python standard
        library.

        Parameters
        ----------
        format : integer, optional
            One of the lzma format codes, e.g., ``lzma.FORMAT_XZ``.
        check : integer, optional
            One of the lzma check codes, e.g., ``lzma.CHECK_NONE``.
        preset : integer, optional
            An integer between 0 and 9 inclusive, specifying the compression
            level.
        filters : list, optional
            A list of dictionaries specifying compression filters. If
            filters are provided, 'preset' must be None.

        """

        codec_id = 'lzma'

        def __init__(
            self,
            format: int = 1,
            check: int = -1,
            preset: int | None = None,
            filters: list[dict[str, Any]] | None = None,
        ) -> None:
            self.format = format
            self.check = check
            self.preset = preset
            self.filters = filters

        def get_config(self) -> LZMAConfig:
            return cast(LZMAConfig, super().get_config())

        def encode(self, buf: BufferLike) -> bytes:
            # normalise inputs
            arr = ensure_contiguous_ndarray(buf)

            # do compression
            compressed: bytes = _lzma_module.compress(
                arr,
                format=self.format,
                check=self.check,
                preset=self.preset,
                filters=self.filters,
            )
            return compressed

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
            dec = _lzma_module.decompress(arr, format=self.format, filters=self.filters)

            # handle destination
            return ndarray_copy(dec, out)

        def __repr__(self) -> str:
            return f'{type(self).__name__}(format={self.format!r}, check={self.check!r}, preset={self.preset!r}, filters={self.filters!r})'
