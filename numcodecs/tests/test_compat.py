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
    typed_bufs = [
        ('B', b'adsdasdas'),
        ('B', bytes(20)),
        ('l', np.arange(100, dtype=np.int64)),
        ('l', array.array('l', b'qwertyuiqwertyui')),
        ('B', mmap.mmap(-1, 10))
    ]
    for typ, buf in typed_bufs:
        b1 = to_buffer(buf)
        assert isinstance(b1, np.ndarray)
        buffer(b1)
        b1mv = memoryview(b1)
        assert b1mv.format is typ
        b2 = buffer_tobytes(buf)
        assert isinstance(b2, bytes)
