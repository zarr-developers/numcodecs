#!/bin/sh
# Build and test numcodecs on 32-bit x86 Linux.
#
# Runs as root inside an i386 Alpine container with the repository mounted at the
# working directory; see .github/workflows/ci-i386.yml. To run it locally:
#
#   docker run --rm --platform linux/386 -v "$PWD:/src" -w /src \
#     alpine:3.24 sh .github/scripts/test-i386.sh
set -eux

apk add --no-cache \
    build-base cargo cmake cython git meson pkgconf rust sed uv zstd \
    python3 python3-dev py3-numpy py3-numpy-dev py3-setuptools py3-wheel

# The checkout belongs to the runner's user, not root; let git (and so
# setuptools-scm) read it anyway.
git config --global --add safe.directory '*'

# numpy publishes no i386 musllinux wheels, and building it from source takes most
# of this job, so use Alpine's py3-numpy. uv does not count packages seen through
# a --system-site-packages venv as installed, so install into the container's
# system Python instead. --only-binary numpy makes uv fail rather than build numpy
# from source if anything asks for a version Alpine does not ship (uv pip ignores
# UV_NO_BUILD_PACKAGE).
export UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1
uv_install() { uv pip install --only-binary numpy "$@"; }
APK_NUMPY=$(python3 -c 'import numpy; print(numpy.__version__)')
check_numpy() {
    python3 - "$APK_NUMPY" <<'PY'
import sys
from importlib.metadata import distribution

import numpy

apk_version = sys.argv[1]
installer = (distribution("numpy").read_text("INSTALLER") or "").strip()
assert numpy.__version__ == apk_version, (numpy.__version__, apk_version)
assert int(apk_version.split(".")[0]) >= 2, apk_version
assert installer != "uv", "numpy was replaced by uv"
print("numpy", numpy.__version__, "from Alpine at", numpy.__file__)
PY
}

uv_install --group dev
check_numpy
PYTHON_INCLUDE=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
PYTHON_LIB=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

# zfpy publishes no 32-bit wheels, so build it from source.
rm -rf zfp
git clone --depth 1 https://github.com/LLNL/zfp
sed -i "s/distutils.sysconfig/sysconfig/g" zfp/python/scikit-build-cmake/FindPythonExtensions.cmake
cmake zfp -B zfp/build -DBUILD_ZFPY=ON -DBUILD_TESTING=OFF \
    -DPYTHON_LIBRARY="$PYTHON_LIB" -DPYTHON_INCLUDE_DIR="$PYTHON_INCLUDE"
make -j "$(nproc)" -C zfp/build
make -C zfp/build install
uv_install --no-build-isolation ./zfp

uv_install --no-build-isolation -v .
uv_install "numcodecs[test,test_extras,msgpack,google_crc32c,crc32c,zfpy]"
uv pip list
check_numpy

python3 -m pytest -v
