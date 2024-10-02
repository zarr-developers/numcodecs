"""
An attempt at replacing bundled versin of blosc with
the blosc-python package, which provides pre-build wheels.

List of functions to deprecate:
[
'destroy',
'init',
'compname_to_compcode',
'cbuffer_sizes',
'cbuffer_metainfo',
'err_bad_cname',
'decompress_partial'
]

List of behaviour to deprecate:
- Passing cname as bytes

"""

from numcodecs.abc import Codec
import numpy as np

import blosc
from blosc import (
    BITSHUFFLE,
    SHUFFLE,
    NOSHUFFLE,
    MAX_BUFFERSIZE,
    MAX_THREADS,
    MAX_TYPESIZE,
    VERSION_STRING,
    VERSION_DATE,
)

__all__ = [
    "BITSHUFFLE",
    "SHUFFLE",
    "NOSHUFFLE",
    "MAX_BUFFERSIZE",
    "MAX_THREADS",
    "MAX_TYPESIZE",
    "VERSION_STRING",
    "VERSION_DATE",
    "list_compressors",
    'get_nthreads',
]

AUTOBLOCKS = 0
AUTOSHUFFLE = -1
_shuffle_repr = ['AUTOSHUFFLE', 'NOSHUFFLE', 'SHUFFLE', 'BITSHUFFLE']


def list_compressors() -> list[str]:
    """Get a list of compressors supported in blosc."""
    return blosc.compressor_list()


def get_nthreads() -> int:
    """
    Get the number of threads that Blosc uses internally for compression and
    decompression.
    """
    nthreads = blosc.set_nthreads(1)
    blosc.set_nthreads(nthreads)
    return nthreads


def set_nthreads(nthreads: int) -> None:
    """
    Set the number of threads that Blosc uses internally for compression and
    decompression.
    """
    blosc.set_nthreads(nthreads)


def cbuffer_complib(source) -> str:
    """Return the name of the compression library used to compress `source`."""
    return blosc.get_clib(source)


def _check_not_object_array(arr):
    if arr.dtype == object:
        raise TypeError("object arrays are not supported")


def _check_buffer_size(buf, max_buffer_size):
    if isinstance(buf, np.ndarray):
        size = buf.nbytes
    else:
        size = len(buf)

    if size > max_buffer_size:
        msg = f"Codec does not support buffers of > {max_buffer_size} bytes"
        raise ValueError(msg)


def compress(source, cname: str, clevel: int, shuffle: int = SHUFFLE, blocksize=AUTOBLOCKS):
    """
    Compress data.

    Parameters
    ----------
    source : bytes-like
        Data to be compressed. Can be any object supporting the buffer
        protocol.
    cname : bytes
        Name of compression library to use.
    clevel : int
        Compression level.
    shuffle : int
        Either NOSHUFFLE (0), SHUFFLE (1), BITSHUFFLE (2) or AUTOSHUFFLE (-1). If AUTOSHUFFLE,
        bit-shuffle will be used for buffers with itemsize 1, and byte-shuffle will
        be used otherwise. The default is `SHUFFLE`.
    blocksize : int
        The requested size of the compressed blocks.  If 0, an automatic blocksize will
        be used.

    Returns
    -------
    dest : bytes
        Compressed data.

    """
    if shuffle == AUTOSHUFFLE:
        if source.itemsize == 1:
            shuffle = BITSHUFFLE
        else:
            shuffle = SHUFFLE
    blosc.set_blocksize(blocksize)
    if isinstance(source, np.ndarray):
        _check_not_object_array(source)
        result = blosc.compress_ptr(
            source.ctypes.data,
            source.size,
            source.dtype.itemsize,
            cname=cname,
            clevel=clevel,
            shuffle=shuffle,
        )
    else:
        result = blosc.compress(source, cname=cname, clevel=clevel, shuffle=shuffle)
    blosc.set_blocksize(AUTOBLOCKS)
    return result


def decompress(source, dest: np.ndarray | bytearray | None = None):
    """
    Decompress data.

    Parameters
    ----------
    source : bytes-like
        Compressed data, including blosc header. Can be any object supporting the buffer
        protocol.
    dest : array-like, optional
        Object to decompress into.

    Returns
    -------
    dest : bytes
        Object containing decompressed data.

    """
    if dest is None:
        return blosc.decompress(source)
    elif isinstance(dest, np.ndarray):
        _check_not_object_array(dest)
        blosc.decompress_ptr(source, dest.ctypes.data)
    else:
        dest[:] = blosc.decompress(source)


class Blosc(Codec):
    """
    Codec providing compression using the Blosc meta-compressor.

    Parameters
    ----------
    cname : string, optional
        A string naming one of the compression algorithms available within blosc, e.g.,
        'zstd', 'blosclz', 'lz4', 'lz4hc', 'zlib' or 'snappy'.
    clevel : integer, optional
        An integer between 0 and 9 specifying the compression level.
    shuffle : integer, optional
        Either NOSHUFFLE (0), SHUFFLE (1), BITSHUFFLE (2) or AUTOSHUFFLE (-1). If AUTOSHUFFLE,
        bit-shuffle will be used for buffers with itemsize 1, and byte-shuffle will
        be used otherwise. The default is `SHUFFLE`.
    blocksize : int
        The requested size of the compressed blocks.  If 0 (default), an automatic
        blocksize will be used.

    See Also
    --------
    numcodecs.zstd.Zstd, numcodecs.lz4.LZ4

    """

    codec_id = 'blosc'
    NOSHUFFLE = NOSHUFFLE
    SHUFFLE = SHUFFLE
    BITSHUFFLE = BITSHUFFLE
    AUTOSHUFFLE = AUTOSHUFFLE
    max_buffer_size = 2**31 - 1

    def __init__(self, cname='lz4', clevel=5, shuffle=SHUFFLE, blocksize=AUTOBLOCKS):
        self.cname = cname
        if isinstance(cname, str):
            self._cname_bytes = cname.encode('ascii')
        else:
            self._cname_bytes = cname
        self.clevel = clevel
        self.shuffle = shuffle
        self.blocksize = blocksize

    def encode(self, buf):
        _check_buffer_size(buf, self.max_buffer_size)
        return compress(
            buf, self.cname, clevel=self.clevel, shuffle=self.shuffle, blocksize=self.blocksize
        )

    def decode(self, buf, out=None):
        _check_buffer_size(buf, self.max_buffer_size)
        return decompress(buf, out)

    def __repr__(self):
        r = f'{type(self).__name__}(cname={self.cname!r}, clevel={self.clevel!r}, shuffle={_shuffle_repr[self.shuffle + 1]}, blocksize={self.blocksize})'
        return r
