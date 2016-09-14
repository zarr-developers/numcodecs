# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from nose.tools import assert_is_instance


from numcodecs.registry import get_codec
from numcodecs.abc import Codec


def test_registry():
    codec = get_codec({'id': 'blosc'})
    assert_is_instance(codec, Codec)
    codec = get_codec({'id': 'zlib'})
    assert_is_instance(codec, Codec)
    codec = get_codec({'id': 'bz2'})
    assert_is_instance(codec, Codec)
