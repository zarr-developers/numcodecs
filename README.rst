Numcodecs
=========

Numcodecs is a Python package providing buffer compression and transformation 
codecs for use in data storage and communication applications.

.. image:: https://readthedocs.org/projects/numcodecs/badge/?version=latest
    :target: https://numcodecs.readthedocs.io/en/latest/?badge=latest

.. image:: https://github.com/zarr-developers/numcodecs/workflows/Linux%20CI/badge.svg?branch=main
    :target: https://github.com/zarr-developers/numcodecs/actions?query=workflow%3A%22Linux+CI%22

.. image:: https://github.com/zarr-developers/numcodecs/workflows/OSX%20CI/badge.svg?branch=main
    :target: https://github.com/zarr-developers/numcodecs/actions?query=workflow%3A%22OSX+CI%22

.. image:: https://github.com/zarr-developers/numcodecs/workflows/Wheels/badge.svg?branch=main
    :target: https://github.com/zarr-developers/numcodecs/actions?query=workflow%3AWheels

.. image:: https://codecov.io/gh/zarr-developers/numcodecs/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/zarr-developers/numcodecs

---
If you already have native Blosc, Zstd, and LZ4 installed on your system and want to use these system libraries instead of the vendored sources, you
should set the `NUMCODECS_USE_SYSTEM_LIBS=1` environment variable when building the wheel, like this:

    $ NUMCODECS_USE_SYSTEM_LIBS=1 pip install numcodecs --no-binary numcodecs

Blosc, Zstd, and LZ4 are found via the `pkg-config` utility. Moreover, you must build all 3 `blosc`, `libzstd`, and `liblz4`
components. C-Blosc comes with full sources for LZ4, LZ4HC, Snappy, Zlib and Zstd and in general, you should not worry about not having (or CMake not finding) the libraries in your system because by default the included sources will be automatically compiled and included in the C-Blosc library. This effectively means that you can be confident in having a complete support for all the codecs in all the Blosc deployments (unless you are explicitly excluding support for some of them). To compile blosc, see these [instructions](https://github.com/Blosc/c-blosc?tab=readme-ov-file#compiling-the-blosc-library).