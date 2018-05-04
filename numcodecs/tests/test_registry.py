# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import pytest


from numcodecs.registry import get_codec


def test_registry_errors():
    with pytest.raises(ValueError):
        get_codec({'id': 'foo'})


def test_get_codec_argument():
    # Check that get_codec doesn't modify its argument.
    arg = {"id": "json"}
    before = dict(arg)
    get_codec(arg)
    assert before == arg
