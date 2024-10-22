from __future__ import annotations

import pytest


def test_zarr3_import():
    ERROR_MESSAGE_MATCH = "zarr 3.0.0 or later.*"

    try:
        import zarr

        if zarr.__version__ < "3.0.0":
            with pytest.raises(ImportError, match=ERROR_MESSAGE_MATCH):
                import numcodecs._zarr3  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=ERROR_MESSAGE_MATCH):
            import numcodecs._zarr3  # noqa: F401
