import ctypes

from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray_like

# constants from libaec.h
AEC_OK = 0
AEC_CONF_ERROR = -1
AEC_STREAM_ERROR = -2
AEC_DATA_ERROR = -3
AEC_MEM_ERROR = -4

# constants from szlib.h
# Only SZ_MSB_OPTION_MASK and SZ_NN_OPTION_MASK are relevant to the libaec implementation
SZ_ALLOW_K13_OPTION_MASK = 1
SZ_CHIP_OPTION_MASK = 2
SZ_EC_OPTION_MASK = 4
SZ_LSB_OPTION_MASK = 8
SZ_MSB_OPTION_MASK = 16
SZ_NN_OPTION_MASK = 32
SZ_RAW_OPTION_MASK = 128

SZ_OK = AEC_OK
SZ_OUTBUFF_FULL = 2

SZ_NO_ENCODER_ERROR = -1
SZ_PARAM_ERROR = AEC_CONF_ERROR
SZ_MEM_ERROR = AEC_MEM_ERROR

SZ_MAX_PIXELS_PER_BLOCK = 32
SZ_MAX_BLOCKS_PER_SCANLINE = 128
SZ_MAX_PIXELS_PER_SCANLINE = SZ_MAX_PIXELS_PER_BLOCK * SZ_MAX_BLOCKS_PER_SCANLINE


class Params(ctypes.Structure):

    _fields_ = [
        ("options_mask", ctypes.c_int),
        ("bits_per_pixel", ctypes.c_int),
        ("pixels_per_block", ctypes.c_int),
        ("pixels_per_scanline", ctypes.c_int)
    ]


libsz = False
for ext in [".so", ".dylib", ".dll"]:
    try:
        libsz = ctypes.cdll.LoadLibrary("libsz" + ext)
    except OSError:
        pass

if libsz:
    BufftoBuffCompress = libsz.SZ_BufftoBuffCompress
    BufftoBuffCompress.restype = ctypes.c_int
    BufftoBuffCompress.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(Params)
    ]

    BufftoBuffDecompress = libsz.SZ_BufftoBuffDecompress
    BufftoBuffDecompress.restype = ctypes.c_int
    BufftoBuffDecompress.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(Params)
    ]


def check_sz():
    if not libsz:  # pragma: no cover
        raise ImportError("libsz could not be loaded, please install libaec")


class HDF5SzipCodec(Codec):
    """The SZIP codec, as implemented in NASA HDF5

    See description:

    The shared library used could in principle be the original SZIP, but it is
    also includes in libaec, which is available on conda-forge and recommended.

    All parameters must be defined.
    """

    codec_id = "hdf5_szip"

    def __init__(self, mask, bits_per_pixel, pixels_per_block, pixels_per_scanline):
        assert pixels_per_block <= SZ_MAX_PIXELS_PER_BLOCK
        assert pixels_per_scanline <= SZ_MAX_PIXELS_PER_SCANLINE
        self.mask = mask
        self.pixels_per_block = pixels_per_block
        self.bits_per_pixel = bits_per_pixel
        self.pixels_per_scanline = pixels_per_scanline

    def decode(self, buf, out=None):
        check_sz()
        buf = memoryview(buf)
        param = Params(
            self.mask, self.bits_per_pixel, self.pixels_per_block, self.pixels_per_scanline
        )
        lout = int.from_bytes(buf[:4], "little")
        dest_len = ctypes.c_size_t(lout)
        p_dest_len = ctypes.pointer(dest_len)
        if out is None:
            out = ctypes.create_string_buffer(lout)
        buf2 = ctypes.c_char_p(buf.obj[4:])
        ok = BufftoBuffDecompress(
            out, p_dest_len, buf2, buf.nbytes - 4, param
        )
        assert dest_len.value == lout  # that we got the expected number of bytes
        assert ok == 0
        return out

    def encode(self, buf):
        check_sz()
        buf = ensure_ndarray_like(buf)
        param = Params(
            self.mask, self.bits_per_pixel, self.pixels_per_block, self.pixels_per_scanline
        )
        buf_nbytes = buf.nbytes
        buf2 = buf.ctypes.data
        out = ctypes.create_string_buffer(buf_nbytes+4)
        # Store input nbytes as first four bytes little endian
        in_len = ctypes.c_int32(buf_nbytes)
        out[:4] = bytes(in_len)
        dest_len = ctypes.c_size_t(buf_nbytes)
        p_dest_len = ctypes.pointer(dest_len)
        ok = BufftoBuffCompress(
            ctypes.addressof(out)+4, p_dest_len, buf2, buf_nbytes, param
        )
        assert ok == 0
        return out[:dest_len.value+4]
