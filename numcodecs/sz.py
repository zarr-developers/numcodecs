import ctypes
from numcodecs.abc import Codec


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
    if not libsz:
        raise ImportError("libsz could not be loaded")


class HdfSzipCodec(Codec):

    codec_id = "hdf_szip"

    def __init__(self, mask, bits_per_pixel, pix_per_block, pix_per_scanline):
        self.mask = mask
        self.pix_per_block = pix_per_block
        self.bits_per_pixel = bits_per_pixel
        self.pix_per_scanline = pix_per_scanline

    def decode(self, buf, out=None):
        check_sz()
        buf = memoryview(buf)
        param = Params(self.mask, self.bits_per_pixel, self.pix_per_block, self.pix_per_scanline)
        lout = int.from_bytes(buf[:4], "little")
        dest_len = ctypes.c_int(lout)
        p_dest_len = ctypes.pointer(dest_len)
        if out is None:
            out = ctypes.create_string_buffer(lout)
        buf2 = ctypes.c_void_p.from_buffer_copy(buf, 4)  # shouldn't need copy, but wants writable
        ok = BufftoBuffDecompress(
            out, p_dest_len, buf2, buf.nbytes - 4, param
        )
        assert dest_len == lout  # that we got the expected number of bytes
        assert ok == 0
        return out

    def encode(self, buf):
        check_sz()
        raise NotImplementedError


def testme():
    sample_buffer = b'\x00\x02\x00\x00\x15UUUUUUUQUUUUUUUU\x15UUUUUUUQUUUUUUUU\x15UUUUUUUQUUUUUUUU\x15UUUUUUUQUUUUUUUU'
    # pl = h5obj.id.get_create_plist().get_filter(0)
    # mask, pix_per_block, bits_per_pixel, pix_per_scanline = pl[2]
    # (141, 32, 16, 256)
    # lout = 512

    codec = HdfSzipCodec(mask=141, pix_per_block=32, bits_per_pixel=16, pix_per_scanline=256)
    codec.decode(sample_buffer)  # Bus Error
