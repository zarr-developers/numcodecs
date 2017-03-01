# -*- coding: utf-8 -*-
# cython: embedsignature=True
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
from __future__ import absolute_import, print_function, division
import threading
import os


from cpython.buffer cimport PyBUF_ANY_CONTIGUOUS, PyBUF_WRITEABLE
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING


from numcodecs.compat_ext cimport Buffer
from numcodecs.compat_ext import Buffer
from numcodecs.compat import PY2, text_type
from numcodecs.abc import Codec


cdef extern from "blosc.h":
    cdef enum:
        BLOSC_MAX_OVERHEAD,
        BLOSC_VERSION_STRING,
        BLOSC_VERSION_DATE,
        BLOSC_NOSHUFFLE,
        BLOSC_SHUFFLE,
        BLOSC_BITSHUFFLE,
        BLOSC_MAX_BUFFERSIZE,
        BLOSC_MAX_THREADS,
        BLOSC_MAX_TYPESIZE

    void blosc_init()
    void blosc_destroy()
    int blosc_get_nthreads()
    int blosc_set_nthreads(int nthreads)
    int blosc_set_compressor(const char *compname)
    char* blosc_list_compressors()
    int blosc_compress(int clevel, int doshuffle, size_t typesize, size_t nbytes, void *src,
                       void *dest, size_t destsize) nogil
    int blosc_decompress(void *src, void *dest, size_t destsize) nogil
    int blosc_compname_to_compcode(const char *compname)
    int blosc_compress_ctx(int clevel, int doshuffle, size_t typesize, size_t nbytes,
                           const void*src, void*dest, size_t destsize, const char*compressor,
                           size_t blocksize, int numinternalthreads) nogil
    int blosc_decompress_ctx(const void *src, void *dest, size_t destsize,
                             int numinternalthreads) nogil
    void blosc_cbuffer_sizes(const void *cbuffer, size_t *nbytes, size_t *cbytes, size_t *blocksize)


MAX_OVERHEAD = BLOSC_MAX_OVERHEAD
MAX_BUFFERSIZE = BLOSC_MAX_BUFFERSIZE
MAX_THREADS = BLOSC_MAX_THREADS
MAX_TYPESIZE = BLOSC_MAX_TYPESIZE
VERSION_STRING = <char *> BLOSC_VERSION_STRING
VERSION_DATE = <char *> BLOSC_VERSION_DATE
if not PY2:
    VERSION_STRING = VERSION_STRING.decode()
    VERSION_DATE = VERSION_DATE.decode()
__version__ = VERSION_STRING
NOSHUFFLE = BLOSC_NOSHUFFLE
SHUFFLE = BLOSC_SHUFFLE
BITSHUFFLE = BLOSC_BITSHUFFLE


def init():
    """Initialize the Blosc library environment."""
    blosc_init()


def destroy():
    """Destroy the Blosc library environment."""
    blosc_destroy()


def compname_to_compcode(cname):
    """Return the compressor code associated with the compressor name. If the compressor name is
    not recognized, or there is not support for it in this build, -1 is returned instead."""
    if isinstance(cname, text_type):
        cname = cname.encode('ascii')
    return blosc_compname_to_compcode(cname)


def list_compressors():
    """Get a list of compressors supported in the current build."""
    return text_type(blosc_list_compressors(), 'ascii').split(',')


def get_nthreads():
    """Get the number of threads that Blosc uses internally for compression and decompression."""
    return blosc_get_nthreads()


def set_nthreads(int nthreads):
    """Set the number of threads that Blosc uses internally for compression and decompression."""
    return blosc_set_nthreads(nthreads)


def cbuffer_sizes(source):
    """Return information about a compressed buffer, namely the number of uncompressed bytes (
    `nbytes`) and compressed (`cbytes`).  It also returns the `blocksize` (which is used
    internally for doing the compression by blocks).

    Returns
    -------
    nbytes : int
    cbytes : int
    blocksize : int

    """
    cdef:
        Buffer buffer
        size_t nbytes, cbytes, blocksize

    # obtain buffer
    buffer = Buffer(source, PyBUF_ANY_CONTIGUOUS)

    # determine buffer size
    blosc_cbuffer_sizes(buffer.ptr, &nbytes, &cbytes, &blocksize)

    # release buffers
    buffer.release()

    return nbytes, cbytes, blocksize


def compress(source, char* cname, int clevel, int shuffle, int blocksize=0):
    """Compress data.

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
        Shuffle filter.
    blocksize : int
        The requested size of the compressed blocks.  If 0, an automatic blocksize will be used.

    Returns
    -------
    dest : bytes
        Compressed data.

    """

    cdef:
        char *source_ptr
        char *dest_ptr
        Buffer source_buffer
        size_t nbytes, cbytes, itemsize
        bytes dest

    # setup source buffer
    source_buffer = Buffer(source, PyBUF_ANY_CONTIGUOUS)
    source_ptr = source_buffer.ptr
    nbytes = source_buffer.nbytes
    itemsize = source_buffer.itemsize

    try:

        # setup destination
        dest = PyBytes_FromStringAndSize(NULL, nbytes + BLOSC_MAX_OVERHEAD)
        dest_ptr = PyBytes_AS_STRING(dest)

        # perform compression
        if _get_use_threads():
            # allow blosc to use threads internally

            # set compressor
            compressor_set = blosc_set_compressor(cname)
            if compressor_set < 0:
                raise ValueError('compressor not supported: %r' % cname)

            # set blocksize
            os.environ['BLOSC_BLOCKSIZE'] = str(blocksize)

            # perform compression
            with nogil:
                cbytes = blosc_compress(clevel, shuffle, itemsize, nbytes, source_ptr, dest_ptr,
                                        nbytes + BLOSC_MAX_OVERHEAD)

            # unset blocksize
            del os.environ['BLOSC_BLOCKSIZE']

        else:
            with nogil:
                cbytes = blosc_compress_ctx(clevel, shuffle, itemsize, nbytes, source_ptr, dest_ptr,
                                            nbytes + BLOSC_MAX_OVERHEAD, cname, blocksize, 1)

    finally:

        # release buffers
        source_buffer.release()

    # check compression was successful
    if cbytes <= 0:
        raise RuntimeError('error during blosc compression: %d' % cbytes)

    # resize after compression
    dest = dest[:cbytes]

    return dest


def decompress(source, dest=None):
    """Decompress data.

    Parameters
    ----------
    source : bytes-like
        Compressed data, including blosc header. Can be any object supporting the buffer protocol.
    dest : array-like, optional
        Object to decompress into.

    Returns
    -------
    dest : bytes
        Object containing decompressed data.

    """
    cdef:
        int ret
        char *source_ptr
        char *dest_ptr
        Buffer source_buffer
        Buffer dest_buffer = None
        size_t nbytes, cbytes, blocksize

    # setup source buffer
    source_buffer = Buffer(source, PyBUF_ANY_CONTIGUOUS)
    source_ptr = source_buffer.ptr

    # determine buffer size
    blosc_cbuffer_sizes(source_ptr, &nbytes, &cbytes, &blocksize)

    # setup destination buffer
    if dest is None:
        # allocate memory
        dest = PyBytes_FromStringAndSize(NULL, nbytes)
        dest_ptr = PyBytes_AS_STRING(dest)
        dest_nbytes = nbytes
    else:
        dest_buffer = Buffer(dest, PyBUF_ANY_CONTIGUOUS | PyBUF_WRITEABLE)
        dest_ptr = dest_buffer.ptr
        dest_nbytes = dest_buffer.nbytes

    try:

        # guard condition
        if dest_nbytes < nbytes:
            raise ValueError('destination buffer too small; expected at least %s, '
                             'got %s' % (nbytes, dest_nbytes))

        # perform decompression
        if _get_use_threads():
            # allow blosc to use threads internally
            with nogil:
                ret = blosc_decompress(source_ptr, dest_ptr, nbytes)
        else:
            with nogil:
                ret = blosc_decompress_ctx(source_ptr, dest_ptr, nbytes, 1)

    finally:

        # release buffers
        source_buffer.release()
        if dest_buffer is not None:
            dest_buffer.release()

    # handle errors
    if ret <= 0:
        raise RuntimeError('error during blosc decompression: %d' % ret)

    return dest


# set the value of this variable to True or False to override the
# default adaptive behaviour
use_threads = None


def _get_use_threads():
    global use_threads

    if use_threads in [True, False]:
        # user has manually overridden the default behaviour
        _use_threads = use_threads

    else:
        # adaptive behaviour: allow blosc to use threads if it is being
        # called from the main Python thread, inferring that it is being run
        # from within a single-threaded program; otherwise do not allow
        # blosc to use threads, inferring it is being run from within a
        # multi-threaded program
        if hasattr(threading, 'main_thread'):
            _use_threads = (threading.main_thread() ==
                            threading.current_thread())
        else:
            _use_threads = threading.current_thread().name == 'MainThread'

    return _use_threads


_shuffle_repr = ['NOSHUFFLE', 'SHUFFLE', 'BITSHUFFLE']


class Blosc(Codec):
    """Codec providing compression using the Blosc meta-compressor.

    Parameters
    ----------
    cname : string, optional
        A string naming one of the compression algorithms available within blosc, e.g., 'zstd',
        'blosclz', 'lz4', 'lz4hc', 'zlib' or 'snappy'.
    clevel : integer, optional
        An integer between 0 and 9 specifying the compression level.
    shuffle : integer, optional
        Either NOSHUFFLE (0), SHUFFLE (1) or BITSHUFFLE (2).
    blocksize : int
        The requested size of the compressed blocks.  If 0 (default), an automatic blocksize will
        be used.

    See Also
    --------
    numcodecs.zstd.Zstd, numcodecs.lz4.LZ4

    """

    codec_id = 'blosc'
    NOSHUFFLE = NOSHUFFLE
    SHUFFLE = SHUFFLE
    BITSHUFFLE = BITSHUFFLE

    def __init__(self, cname='lz4', clevel=5, shuffle=1, blocksize=0):
        self.cname = cname
        if isinstance(cname, text_type):
            self._cname_bytes = cname.encode('ascii')
        else:
            self._cname_bytes = cname
        self.clevel = clevel
        self.shuffle = shuffle
        self.blocksize = blocksize

    def encode(self, buf):
        return compress(buf, self._cname_bytes, self.clevel, self.shuffle, self.blocksize)

    def decode(self, buf, out=None):
        return decompress(buf, out)

    def __repr__(self):
        r = '%s(cname=%r, clevel=%r, shuffle=%s, blocksize=%s)' % \
            (type(self).__name__,
             self.cname,
             self.clevel,
             _shuffle_repr[self.shuffle],
             self.blocksize)
        return r
