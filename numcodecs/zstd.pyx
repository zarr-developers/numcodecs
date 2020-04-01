# cython: embedsignature=True
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# cython: language_level=3


from cpython.buffer cimport PyBUF_ANY_CONTIGUOUS, PyBUF_WRITEABLE
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING


from .compat_ext cimport Buffer
from .compat_ext import Buffer
from .compat import ensure_contiguous_ndarray
from .abc import Codec


cdef extern from "zstd.h":

    unsigned ZSTD_versionNumber() nogil

    size_t ZSTD_compress(void* dst,
                         size_t dstCapacity,
                         const void* src,
                         size_t srcSize,
                         int compressionLevel) nogil

    size_t ZSTD_decompress(void* dst,
                           size_t dstCapacity,
                           const void* src,
                           size_t compressedSize) nogil

    unsigned long long ZSTD_getDecompressedSize(const void* src,
                                                size_t srcSize) nogil

    int ZSTD_maxCLevel() nogil

    size_t ZSTD_compressBound(size_t srcSize) nogil

    unsigned ZSTD_isError(size_t code) nogil

    const char* ZSTD_getErrorName(size_t code)


VERSION_NUMBER = ZSTD_versionNumber()
MAJOR_VERSION_NUMBER = VERSION_NUMBER // (100 * 100)
MINOR_VERSION_NUMBER = (VERSION_NUMBER - (MAJOR_VERSION_NUMBER * 100 * 100)) // 100
MICRO_VERSION_NUMBER = (
    VERSION_NUMBER -
    (MAJOR_VERSION_NUMBER * 100 * 100) -
    (MINOR_VERSION_NUMBER * 100)
)
__version__ = '%s.%s.%s' % (MAJOR_VERSION_NUMBER, MINOR_VERSION_NUMBER, MICRO_VERSION_NUMBER)
DEFAULT_CLEVEL = 1
MAX_CLEVEL = ZSTD_maxCLevel()


def compress(source, int level=DEFAULT_CLEVEL):
    """Compress data.

    Parameters
    ----------
    source : bytes-like
        Data to be compressed. Can be any object supporting the buffer
        protocol.
    level : int
        Compression level (1-22).

    Returns
    -------
    dest : bytes
        Compressed data.
    """

    cdef:
        char *source_ptr
        char *dest_ptr
        Buffer source_buffer
        size_t source_size, dest_size, compressed_size
        bytes dest

    # check level
    if level <= 0:
        level = DEFAULT_CLEVEL
    if level > MAX_CLEVEL:
        level = MAX_CLEVEL

    # setup source buffer
    source_buffer = Buffer(source, PyBUF_ANY_CONTIGUOUS)
    source_ptr = source_buffer.ptr
    source_size = source_buffer.nbytes

    try:

        # setup destination
        dest_size = ZSTD_compressBound(source_size)
        dest = PyBytes_FromStringAndSize(NULL, dest_size)
        dest_ptr = PyBytes_AS_STRING(dest)

        # perform compression
        with nogil:
            compressed_size = ZSTD_compress(dest_ptr, dest_size, source_ptr, source_size, level)

    finally:

        # release buffers
        source_buffer.release()

    # check compression was successful
    if ZSTD_isError(compressed_size):
        error = ZSTD_getErrorName(compressed_size)
        raise RuntimeError('Zstd compression error: %s' % error)

    # resize after compression
    dest = dest[:compressed_size]

    return dest


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
    cdef:
        char *source_ptr
        char *dest_ptr
        Buffer source_buffer
        Buffer dest_buffer = None
        size_t source_size, dest_size, decompressed_size

    # setup source buffer
    source_buffer = Buffer(source, PyBUF_ANY_CONTIGUOUS)
    source_ptr = source_buffer.ptr
    source_size = source_buffer.nbytes

    try:

        # determine uncompressed size
        dest_size = ZSTD_getDecompressedSize(source_ptr, source_size)
        if dest_size == 0:
            raise RuntimeError('Zstd decompression error: invalid input data')

        # setup destination buffer
        if dest is None:
            # allocate memory
            dest = PyBytes_FromStringAndSize(NULL, dest_size)
            dest_ptr = PyBytes_AS_STRING(dest)
        else:
            arr = ensure_contiguous_ndarray(dest)
            dest_buffer = Buffer(arr, PyBUF_ANY_CONTIGUOUS | PyBUF_WRITEABLE)
            dest_ptr = dest_buffer.ptr
            if dest_buffer.nbytes < dest_size:
                raise ValueError('destination buffer too small; expected at least %s, '
                                 'got %s' % (dest_size, dest_buffer.nbytes))

        # perform decompression
        with nogil:
            decompressed_size = ZSTD_decompress(dest_ptr, dest_size, source_ptr, source_size)

    finally:

        # release buffers
        source_buffer.release()
        if dest_buffer is not None:
            dest_buffer.release()

    # check decompression was successful
    if ZSTD_isError(decompressed_size):
        error = ZSTD_getErrorName(decompressed_size)
        raise RuntimeError('Zstd decompression error: %s' % error)
    elif decompressed_size != dest_size:
        raise RuntimeError('Zstd decompression error: expected to decompress %s, got %s' %
                           (dest_size, decompressed_size))

    return dest


class Zstd(Codec):
    """Codec providing compression using Zstandard.

    Parameters
    ----------
    level : int
        Compression level (1-22).

    See Also
    --------
    numcodecs.lz4.LZ4, numcodecs.blosc.Blosc

    """

    codec_id = 'zstd'
    
    # Note: unlike the LZ4 and Blosc codecs, there does not appear to be a (currently)
    # practical limit on the size of buffers that Zstd can process and so we don't 
    # enforce a max_buffer_size option here.

    def __init__(self, level=DEFAULT_CLEVEL):
        self.level = level

    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf)
        return compress(buf, self.level)

    def decode(self, buf, out=None):
        buf = ensure_contiguous_ndarray(buf)
        return decompress(buf, out)

    def __repr__(self):
        r = '%s(level=%r)' % \
            (type(self).__name__,
             self.level)
        return r
