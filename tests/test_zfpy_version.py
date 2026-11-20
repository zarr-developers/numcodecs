# These tests do not need zfpy installed, so they live apart from test_zfpy.py,
# which skips the whole module when the ZFPY codec is unavailable.
import pytest

from numcodecs.zfpy import _zfpy_version_supported


@pytest.mark.parametrize(
    ("zfpy_version", "expected"),
    [
        ("1.0.1", True),
        ("1.0.1.post0", True),  # gh-892: post-release must not break `import numcodecs`
        ("1.0.2", True),
        ("1.1.0rc1", True),
        ("1.0.2+local", True),
        ("1.0.0", False),
        ("1.0.0.post1", False),
        ("1.0.1.dev0", False),
        ("0.5.5", False),
        ("not-a-version", False),
    ],
)
def test_zfpy_version_supported(zfpy_version: str, expected: bool) -> None:
    assert _zfpy_version_supported(zfpy_version) is expected
