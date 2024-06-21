from .abc import Codec
from .compat import ensure_contiguous_ndarray

import blosc2

NOSHUFFLE = 0
SHUFFLE = 1
BITSHUFFLE = 2
AUTOSHUFFLE = -1
AUTOBLOCKS = 0

_shuffles = [blosc2.Filter.NOFILTER, blosc2.Filter.SHUFFLE, blosc2.Filter.BITSHUFFLE]
_shuffle_repr = ['AUTOSHUFFLE', 'NOSHUFFLE', 'SHUFFLE', 'BITSHUFFLE']

cbuffer_sizes = blosc2.get_cbuffer_sizes

def list_compressors():
    return [str(codec).lower().replace("codec.", "") for codec in blosc2.compressor_list()]

def cbuffer_complib(source):
    """Return the name of the compression library used to compress `source`."""
    return blosc2.get_clib(source)

def compress(source, cname: bytes, clevel, shuffle: int=SHUFFLE, blocksize=AUTOBLOCKS):
    cname = cname.decode('ascii')
    blosc2.set_blocksize(blocksize)
    return blosc2.compress(source, codec=getattr(blosc2.Codec, cname.upper()), clevel=clevel, filter=_shuffles[shuffle])


class Blosc(Codec):
    """Codec providing compression using the Blosc meta-compressor.

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
        self.clevel = clevel
        self.shuffle = shuffle
        self.blocksize = blocksize

    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
        return compress(buf, bytes(self.cname, 'ascii'), self.clevel, self.shuffle, self.blocksize)

    def decode(self, buf, out=None):
        buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
        return blosc2.decompress(buf, dst=out)

    # def decode_partial(self, buf, int start, int nitems, out=None):
    #     '''**Experimental**'''
    #     buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
    #     return decompress_partial(buf, start, nitems, dest=out)

    def __repr__(self):
        r = '%s(cname=%r, clevel=%r, shuffle=%s, blocksize=%s)' % \
            (type(self).__name__,
             self.cname,
             self.clevel,
             _shuffle_repr[self.shuffle + 1],
             self.blocksize)
        return r
