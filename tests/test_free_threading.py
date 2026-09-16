import subprocess
import sys
import sysconfig

import pytest

pytestmark = pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="requires a free-threaded Python build",
)


def test_import_does_not_reenable_gil():
    # Run in a fresh interpreter so that no other extension module (pytest plugins,
    # optional dependencies) can have re-enabled the GIL already.
    code = "import sys, numcodecs; sys.exit(int(sys._is_gil_enabled()))"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        "importing numcodecs re-enabled the GIL on a free-threaded build\n" + proc.stderr
    )
