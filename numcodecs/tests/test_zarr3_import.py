from __future__ import annotations

import pytest


def test_zarr3_import():
    ERROR_MESSAGE_MATCH = "zarr 3.0.0 or later.*"

    try:
        import zarr  # noqa: F401
    except ImportError:  # pragma: no cover
        with pytest.raises(ImportError, match=ERROR_MESSAGE_MATCH):
            import numcodecs.zarr3  # noqa: F401
