import pytest

import numpy as np

from numcodecs.sz import HdfSzipCodec, libsz


@pytest.mark.skipif(not libsz, reason="no libaec")
def test_canonical():
    sample_buffer = b'\x00\x02\x00\x00\x15UUUUUUUQUUUUUUUU\x15UUUUUUUQUUUUUUUU' \
                    b'\x15UUUUUUUQUUUUUUUU\x15UUUUUUUQUUUUUUUU'
    # pl = h5obj.id.get_create_plist().get_filter(0)
    # mask, pix_per_block, bits_per_pixel, pix_per_scanline = pl[2]
    # (141, 32, 16, 256)
    # lout = 512

    codec = HdfSzipCodec(mask=141, pix_per_block=32, bits_per_pixel=16, pix_per_scanline=256)
    out = codec.decode(sample_buffer)  # Bus Error
    arr = np.frombuffer(out, dtype="uint16")
    assert (arr == 1).all()

    comp_buff = codec.encode(arr)
    assert comp_buff == sample_buffer


@pytest.mark.skipif(not libsz, reason="no libaec")
@pytest.mark.parametrize(
    "shape", [(10, 100), (10000,)]
)
@pytest.mark.parametrize(
    "dtype", ["uint16", "int32", "int64"]
)
def test_random(shape, dtype):
    arr = np.random.randint(0, 40, size=np.prod(shape), dtype=dtype).reshape(shape)
    codec = HdfSzipCodec(mask=141, pix_per_block=32, bits_per_pixel=16, pix_per_scanline=256)
    buff = codec.encode(arr)
    buff2 = codec.decode(buff)
    arr2 = np.frombuffer(buff2, dtype=dtype).reshape(shape)
    assert (arr == arr2).all()
