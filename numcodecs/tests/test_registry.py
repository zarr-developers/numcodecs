# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from nose.tools import assert_raises


from numcodecs.registry import get_codec


def test_registry_errors():
    with assert_raises(ValueError):
        get_codec({'id': 'foo'})
