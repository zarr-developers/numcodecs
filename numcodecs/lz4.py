from .compat import ensure_contiguous_ndarray
from .abc import Codec

import lz4
import lz4.block as lz4_b


VERSION_STRING = lz4.library_version_string()
__version__ = VERSION_STRING
DEFAULT_ACCELERATION = 1


def compress(source, acceleration :int=DEFAULT_ACCELERATION):
    """Compress data.

    Parameters
    ----------
    source : bytes-like
        Data to be compressed. Can be any object supporting the buffer
        protocol.
    acceleration : int
        Acceleration level. The larger the acceleration value, the faster the algorithm, but also
        the lesser the compression.

    Returns
    -------
    dest : bytes
        Compressed data.

    Notes
    -----
    The compressed output includes a 4-byte header storing the original size of the decompressed
    data as a little-endian 32-bit integer.

    """

    # check level
    if acceleration <= 0:
        acceleration = DEFAULT_ACCELERATION
        # FIXME

    return lz4_b.compress(source)


def decompress(source, dest=None):
    """Decompress data.

    Parameters
    ----------
    source : bytes-like
        Compressed data. Can be any object supporting the buffer protocol.
    dest : array-like, optional
        Object to decompress into.

    Returns
    -------
    dest : bytes
        Object containing decompressed data.

    """
    if dest is not None:
        raise Exception("FIXME")
    return lz4_b.decompress(source)


class LZ4(Codec):
    """Codec providing compression using LZ4.

    Parameters
    ----------
    acceleration : int
        Acceleration level. The larger the acceleration value, the faster the algorithm, but also
        the lesser the compression.

    See Also
    --------
    numcodecs.zstd.Zstd, numcodecs.blosc.Blosc

    """

    codec_id = 'lz4'
    max_buffer_size = 0x7E000000

    def __init__(self, acceleration=DEFAULT_ACCELERATION):
        self.acceleration = acceleration

    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
        return compress(buf, self.acceleration)

    def decode(self, buf, out=None):
        buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
        return decompress(buf, out)

    def __repr__(self):
        r = '%s(acceleration=%r)' % \
            (type(self).__name__,
             self.acceleration)
        return r
