import warnings
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

from packaging.version import InvalidVersion, Version

_MIN_ZFPY_VERSION = Version("1.0.1")  # first release that supports NumPy >= 2


def _zfpy_version_supported(zfpy_version: str) -> bool:
    """Return whether ``zfpy_version`` is a parseable version >= 1.0.1."""
    try:
        return Version(zfpy_version) >= _MIN_ZFPY_VERSION
    except InvalidVersion:
        return False


_zfpy: ModuleType | None = None

_zfpy_version: str | None = None
with suppress(PackageNotFoundError):
    _zfpy_version = version("zfpy")

if _zfpy_version is not None:
    if _zfpy_version_supported(_zfpy_version):
        with suppress(ImportError):
            import zfpy as _zfpy  # type: ignore[no-redef]
    else:  # pragma: no cover
        warnings.warn(
            f"zfpy {_zfpy_version!r} is installed, but the ZFPY codec requires "
            f"zfpy >= {_MIN_ZFPY_VERSION}. The ZFPY codec will be unavailable.",
            UserWarning,
            stacklevel=2,
        )

if _zfpy:
    import numpy as np

    from .abc import Codec
    from .compat import ensure_bytes, ensure_contiguous_ndarray, ndarray_copy

    # noinspection PyShadowingBuiltins
    class ZFPY(Codec):
        """Codec providing compression using zfpy via the Python standard
        library.

        Parameters
        ----------
        mode : integer
            One of the zfpy mode choice, e.g., ``zfpy.mode_fixed_accuracy``.
        tolerance : double, optional
            A double-precision number, specifying the compression accuracy needed.
        rate : double, optional
            A double-precision number, specifying the compression rate needed.
        precision : int, optional
            A integer number, specifying the compression precision needed.

        """

        codec_id = "zfpy"

        def __init__(
            self,
            mode=_zfpy.mode_fixed_accuracy,
            tolerance=-1,
            rate=-1,
            precision=-1,
            compression_kwargs=None,
        ):
            self.mode = mode
            if mode == _zfpy.mode_fixed_accuracy:
                self.compression_kwargs = {"tolerance": tolerance}
            elif mode == _zfpy.mode_fixed_rate:
                self.compression_kwargs = {"rate": rate}
            elif mode == _zfpy.mode_fixed_precision:
                self.compression_kwargs = {"precision": precision}

            self.tolerance = tolerance
            self.rate = rate
            self.precision = precision

        def encode(self, buf):
            # not flatten c-order array and raise exception for f-order array
            if not isinstance(buf, np.ndarray):
                raise TypeError(
                    "The zfp codec does not support none numpy arrays."
                    f" Your buffers were {type(buf)}."
                )
            if buf.flags.c_contiguous:
                flatten = False
            else:
                raise ValueError(
                    "The zfp codec does not support F order arrays. "
                    f"Your arrays flags were {buf.flags}."
                )
            buf = ensure_contiguous_ndarray(buf, flatten=flatten)

            # do compression
            return _zfpy.compress_numpy(buf, write_header=True, **self.compression_kwargs)

        def decode(self, buf, out=None):
            # normalise inputs
            buf = ensure_bytes(buf)
            if out is not None:
                out = ensure_contiguous_ndarray(out)

            # do decompression
            dec = _zfpy.decompress_numpy(buf)

            # handle destination
            if out is not None:
                return ndarray_copy(dec, out)
            else:
                return dec

        def __repr__(self):
            return (
                f"{type(self).__name__}(mode={self.mode!r}, "
                f"tolerance={self.tolerance}, rate={self.rate}, "
                f"precision={self.precision})"
            )
