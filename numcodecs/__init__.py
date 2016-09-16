# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division


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
    from numcodecs.blosc import Blosc
    register_codec(Blosc)
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
