# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import array
import mmap

import pytest

import numpy as np


from numcodecs.compat import buffer, to_buffer, buffer_tobytes


def test_buffer_coercion_raises():
    a = np.array([u'Xin chào thế giới'], dtype=object)
    for e in [a, memoryview(a)]:
        for f in [to_buffer, buffer_tobytes]:
            with pytest.raises(ValueError):
                f(e)


def test_buffer_readonly():
    a = np.arange(100)
    a.setflags(write=True)

    b = to_buffer(a)
    m = memoryview(b)

    assert m.readonly


def test_buffer_coercion():
    bufs = [
        b'adsdasdas',
        bytes(20),
        np.arange(100, dtype=np.int64),
        array.array('l', b'qwertyuiqwertyui'),
        mmap.mmap(-1, 10)
    ]
    for buf in bufs:
        b1 = to_buffer(buf)
        buffer(b1)
        memoryview(b1)
        assert isinstance(b1, np.ndarray)
        b2 = buffer_tobytes(buf)
        assert isinstance(b2, bytes)
