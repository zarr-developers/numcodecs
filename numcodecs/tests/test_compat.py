# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import array
import mmap

import pytest

import numpy as np


from numcodecs.compat import buffer_tobytes


def test_buffer_tobytes_raises():
    a = np.array([u'Xin chào thế giới'], dtype=object)
    with pytest.raises(ValueError):
        buffer_tobytes(a)
    with pytest.raises(ValueError):
        buffer_tobytes(memoryview(a))


def test_buffer_tobytes():
    bufs = [
        b'adsdasdas',
        bytes(20),
        np.arange(100),
        array.array('l', b'qwertyuiqwertyui'),
        mmap.mmap(-1, 10)
    ]
    for buf in bufs:
        b = buffer_tobytes(buf)
        assert isinstance(b, bytes)
