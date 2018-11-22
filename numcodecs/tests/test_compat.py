# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import array


import numpy as np


from numcodecs.compat import ensure_bytes


def test_ensure_bytes():
    bufs = [
        b'adsdasdas',
        bytes(20),
        np.arange(100),
        array.array('l', b'qwertyuiqwertyui')
    ]
    for buf in bufs:
        b = ensure_bytes(buf)
        assert isinstance(b, bytes)
