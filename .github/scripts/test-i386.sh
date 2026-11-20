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
    python3 python3-dev py3-numpy py3-numpy-dev py3-pip py3-pytest

# The checkout belongs to the runner's user, not root; let git (and so
# setuptools-scm) read it anyway.
git config --global --add safe.directory '*'

uv venv
. .venv/bin/activate
uv pip install --group dev
PYTHON_INCLUDE=$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
PYTHON_LIB=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

# zfpy publishes no 32-bit wheels, so build it from source.
rm -rf zfp
git clone --depth 1 https://github.com/LLNL/zfp
sed -i "s/distutils.sysconfig/sysconfig/g" zfp/python/scikit-build-cmake/FindPythonExtensions.cmake
cmake zfp -B zfp/build -DBUILD_ZFPY=ON -DBUILD_TESTING=OFF \
    -DPYTHON_LIBRARY="$PYTHON_LIB" -DPYTHON_INCLUDE_DIR="$PYTHON_INCLUDE"
make -j "$(nproc)" -C zfp/build
make -C zfp/build install
uv pip install ./zfp

uv pip install --no-build-isolation -v .
uv pip install "numcodecs[test,test_extras,msgpack,google_crc32c,crc32c,zfpy]"
uv pip list

python -m pytest -v
