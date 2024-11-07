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

    struct ZSTD_CCtx_s:
        pass
    ctypedef ZSTD_CCtx_s ZSTD_CCtx
    cdef enum ZSTD_cParameter:
        ZSTD_c_compressionLevel=100
        ZSTD_c_checksumFlag=201

    ZSTD_CCtx* ZSTD_createCCtx() nogil
    size_t ZSTD_freeCCtx(ZSTD_CCtx* cctx) nogil
    size_t ZSTD_CCtx_setParameter(ZSTD_CCtx* cctx,
                                  ZSTD_cParameter param,
                                  int value) nogil

    size_t ZSTD_compress2(ZSTD_CCtx* cctx,
                          void* dst,
                          size_t dstCapacity,
                          const void* src,
                          size_t srcSize) nogil

    size_t ZSTD_decompress(void* dst,
                           size_t dstCapacity,
                           const void* src,
                           size_t compressedSize) nogil

    cdef long ZSTD_CONTENTSIZE_UNKNOWN
    cdef long ZSTD_CONTENTSIZE_ERROR
    unsigned long long ZSTD_getFrameContentSize(const void* src,
                                                size_t srcSize) nogil

    int ZSTD_minCLevel() nogil
    int ZSTD_maxCLevel() nogil
    int ZSTD_defaultCLevel() nogil

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
DEFAULT_CLEVEL = 0
MAX_CLEVEL = ZSTD_maxCLevel()


def compress(source, int level=DEFAULT_CLEVEL, bint checksum=False):
    """Compress data.

    Parameters
    ----------
    source : bytes-like
        Data to be compressed. Can be any object supporting the buffer
        protocol.
    level : int
        Compression level (-131072 to 22).
    checksum : bool
        Flag to enable checksums. The default is False.

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
    if level > MAX_CLEVEL:
        level = MAX_CLEVEL

    # setup source buffer
    source_buffer = Buffer(source, PyBUF_ANY_CONTIGUOUS)
    source_ptr = source_buffer.ptr
    source_size = source_buffer.nbytes

    cctx = ZSTD_createCCtx()
    param_set_result = ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level)

    if ZSTD_isError(param_set_result):
        error = ZSTD_getErrorName(param_set_result)
        raise RuntimeError('Could not set zstd compression level: %s' % error)

    param_set_result = ZSTD_CCtx_setParameter(cctx, ZSTD_c_checksumFlag, 1 if checksum else 0)

    if ZSTD_isError(param_set_result):
        error = ZSTD_getErrorName(param_set_result)
        raise RuntimeError('Could not set zstd checksum flag: %s' % error)

    try:

        # setup destination
        dest_size = ZSTD_compressBound(source_size)
        dest = PyBytes_FromStringAndSize(NULL, dest_size)
        dest_ptr = PyBytes_AS_STRING(dest)

        # perform compression
        with nogil:
            compressed_size = ZSTD_compress2(cctx, dest_ptr, dest_size, source_ptr, source_size)

    finally:
        if cctx:
            ZSTD_freeCCtx(cctx)
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
        dest_size = ZSTD_getFrameContentSize(source_ptr, source_size)
        if dest_size == 0 or dest_size == ZSTD_CONTENTSIZE_UNKNOWN or dest_size == ZSTD_CONTENTSIZE_ERROR:
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
        Compression level (-131072 to 22).
    checksum : bool
        Flag to enable checksums. The default is False.

    See Also
    --------
    numcodecs.lz4.LZ4, numcodecs.blosc.Blosc

    """

    codec_id = 'zstd'

    # Note: unlike the LZ4 and Blosc codecs, there does not appear to be a (currently)
    # practical limit on the size of buffers that Zstd can process and so we don't
    # enforce a max_buffer_size option here.

    def __init__(self, level=DEFAULT_CLEVEL, checksum=False):
        self.level = level
        self.checksum = checksum

    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf)
        return compress(buf, self.level, self.checksum)

    def decode(self, buf, out=None):
        buf = ensure_contiguous_ndarray(buf)
        return decompress(buf, out)

    def __repr__(self):
        r = '%s(level=%r)' % \
            (type(self).__name__,
             self.level)
        return r

    @classmethod
    def default_level(cls):
        """Returns the default compression level of the underlying zstd library."""
        return ZSTD_defaultCLevel()

    @classmethod
    def min_level(cls):
        """Returns the minimum compression level of the underlying zstd library."""
        return ZSTD_minCLevel()

    @classmethod
    def max_level(cls):
        """Returns the maximum compression level of the underlying zstd library."""
        return ZSTD_maxCLevel()
