# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import pytest


from numcodecs.registry import get_codec


def test_registry_errors():
    with pytest.raises(ValueError):
        get_codec({'id': 'foo'})
