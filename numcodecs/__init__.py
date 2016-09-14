# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division

from numcodecs.version import version as __version__
from numcodecs.registry import get_codec
try:
    from numcodecs.blosc import Blosc
except ImportError:
    pass
