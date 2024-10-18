from __future__ import annotations
import pytest


def test_zarr3_import():
    try:
        import zarr
    except ImportError:
        pass

    if zarr is None or zarr.__version__ < "3.0.0":
        with pytest.raises(ImportError):
            import numcodecs._zarr3  # noqa: F401
