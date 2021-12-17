import numpy as np

import pytest

from numcodecs.bitround import BitRound
from numcodecs.tests.common import check_encode_decode, check_config, \
    check_repr, check_backwards_compatibility

# adapted from https://github.com/milankl/BitInformation.jl/blob/main/test/round_nearest.jl


@pytest.fixture(params=[np.float32])
def dtype(request):
    return request.param


def test_smoke(dtype):
    data = np.zeros((3, 2), dtype=dtype)
    codec = BitRound(keepbits=10)
    codec.decode(codec.encode(data))
