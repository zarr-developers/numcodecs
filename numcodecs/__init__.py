# -*- coding: utf-8 -*-
# flake8: noqa
"""Numcodecs is a Python package providing buffer compression and
transformation codecs for use in data storage and communication
applications. These include:

* Compression codecs, e.g., Zlib, BZ2, LZMA and Blosc.
* Pre-compression filters, e.g., Delta, Quantize, FixedScaleOffset,
  PackBits, Categorize.
* Integrity checks, e.g., CRC32, Adler32.

All codecs implement the same API, allowing codecs to be organized into
pipelines in a variety of ways.

If you have a question, find a bug, would like to make a suggestion or
contribute code, please `raise an issue on GitHub
<https://github.com/alimanfoo/numcodecs/issues>`_.

"""

from __future__ import absolute_import, print_function, division
import multiprocessing
import atexit


from numcodecs.version import version as __version__
from numcodecs.registry import get_codec, register_codec
from numcodecs.compat import PY2

from numcodecs.zlib import Zlib
register_codec(Zlib)

from numcodecs.bz2 import BZ2
register_codec(BZ2)

if not PY2:
    from numcodecs.lzma import LZMA
    register_codec(LZMA)

try:
    from numcodecs import blosc as _blosc
    from numcodecs.blosc import Blosc
    register_codec(Blosc)
    # initialize blosc
    ncores = multiprocessing.cpu_count()
    _blosc.init()
    _blosc.set_nthreads(min(8, ncores))
    atexit.register(_blosc.destroy)
except ImportError:  # pragma: no cover
    pass

from numcodecs.delta import Delta
register_codec(Delta)

from numcodecs.fixedscaleoffset import FixedScaleOffset
register_codec(FixedScaleOffset)

from numcodecs.packbits import PackBits
register_codec(PackBits)

from numcodecs.categorize import Categorize
register_codec(Categorize)


from numcodecs.checksum32 import CRC32, Adler32
register_codec(CRC32)
register_codec(Adler32)
