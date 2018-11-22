# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import array
import mmap


import numpy as np
import pytest


from numcodecs.compat import (ensure_bytes, ensure_ndarray_from_memory,
                              ensure_memoryview, PY2)
if PY2:  # pragma: py3 no cover
    from numcodecs.compat import ensure_buffer


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


def test_memory_sharing():
    # test ensure_ndarray, ensure_memoryview and ensure_buffer
    typed_bufs = [
        ('u', 1, b'adsdasdas'),
        ('u', 1, bytes(20)),
        ('i', 8, np.arange(100, dtype=np.int64)),
        ('f', 8, np.linspace(0, 1, 100, dtype=np.float64)),
        ('i', 4, array.array('i', b'qwertyuiqwertyui')),
        ('i', 8, array.array('l', b'qwertyuiqwertyui')),
        ('u', 4, array.array('I', b'qwertyuiqwertyui')),
        ('u', 8, array.array('L', b'qwertyuiqwertyui')),
        ('f', 4, array.array('f', b'qwertyuiqwertyui')),
        ('f', 8, array.array('d', b'qwertyuiqwertyui')),
        ('u', 1, mmap.mmap(-1, 10))
    ]
    for typ, siz, buf in typed_bufs:
        a = ensure_ndarray_from_memory(buf)
        assert isinstance(a, np.ndarray)
        if PY2 and isinstance(buf, array.array):
            # array.array does not expose buffer interface on PY2 so type information
            # is not propagated correctly, so skip array.array on PY2
            pass
        else:
            assert a.dtype.kind is typ, buf
            assert a.dtype.itemsize is siz
        if PY2:  # pragma: py3 no cover
            assert np.shares_memory(a, buffer(buf))  # noqa
            b = ensure_buffer(buf)
            assert np.shares_memory(b, buffer(buf))  # noqa
        else:  # pragma: py2 no cover
            assert np.shares_memory(a, memoryview(buf))
            m = ensure_memoryview(buf)
            assert np.shares_memory(m, memoryview(buf))


def test_object_array_raises():
    a = np.array([u'Xin chào thế giới'], dtype=object)
    for e in [a, memoryview(a)]:
        with pytest.raises(ValueError):
            ensure_ndarray_from_memory(e)
        with pytest.raises(ValueError):
            ensure_memoryview(e)
        if PY2:  # pragma: py3 no cover
            with pytest.raises(ValueError):
                ensure_buffer(e)
    with pytest.raises(TypeError):
        ensure_ndarray_from_memory(a.tolist())
    with pytest.raises(TypeError):
        ensure_memoryview(a.tolist())
    if PY2:  # pragma: py3 no cover
        with pytest.raises(TypeError):
            ensure_buffer(a.tolist())


def test_memoryview_writable():
    for writable in [False, True]:
        a = np.arange(100)
        a.setflags(write=writable)
        m = ensure_memoryview(a)
        assert m.readonly != writable
