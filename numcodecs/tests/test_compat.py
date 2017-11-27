# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import array


import numpy as np


from numcodecs.compat import buffer_tobytes


def test_buffer_tobytes():
    bufs = [
        b'adsdasdas',
        bytes(20),
        np.arange(100),
        array.array('l', b'qwertyuiqwertyui')
    ]
    for buf in bufs:
        b = buffer_tobytes(buf)
        assert isinstance(b, bytes)
