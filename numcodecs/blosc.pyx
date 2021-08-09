# cython: embedsignature=True
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# cython: language_level=3
import threading
import multiprocessing
import os


from cpython.buffer cimport PyBUF_ANY_CONTIGUOUS, PyBUF_WRITEABLE
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING


from .compat_ext cimport Buffer
from .compat_ext import Buffer
from .compat import ensure_contiguous_ndarray
from .abc import Codec


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
        BLOSC_MAX_TYPESIZE,
        BLOSC_DOSHUFFLE,
        BLOSC_DOBITSHUFFLE,
        BLOSC_MEMCPYED

    void blosc_init()
    void blosc_destroy()
    int blosc_get_nthreads()
    int blosc_set_nthreads(int nthreads)
    int blosc_set_compressor(const char *compname)
    void blosc_set_blocksize(size_t blocksize)
    char* blosc_list_compressors()
    int blosc_compress(int clevel, int doshuffle, size_t typesize, size_t nbytes,
                       void* src, void* dest, size_t destsize) nogil
    int blosc_decompress(void *src, void *dest, size_t destsize) nogil
    int blosc_getitem(void* src, int start, int nitems, void* dest)
    int blosc_compname_to_compcode(const char* compname)
    int blosc_compress_ctx(int clevel, int doshuffle, size_t typesize, size_t nbytes,
                           const void* src, void* dest, size_t destsize,
                           const char* compressor, size_t blocksize,
                           int numinternalthreads) nogil
    int blosc_decompress_ctx(const void* src, void* dest, size_t destsize,
                             int numinternalthreads) nogil
    void blosc_cbuffer_sizes(const void* cbuffer, size_t* nbytes, size_t* cbytes,
                             size_t* blocksize)
    char* blosc_cbuffer_complib(const void* cbuffer)
    void blosc_cbuffer_metainfo(const void* cbuffer, size_t* typesize, int* flags)


MAX_OVERHEAD = BLOSC_MAX_OVERHEAD
MAX_BUFFERSIZE = BLOSC_MAX_BUFFERSIZE
MAX_THREADS = BLOSC_MAX_THREADS
MAX_TYPESIZE = BLOSC_MAX_TYPESIZE
VERSION_STRING = <char *> BLOSC_VERSION_STRING
VERSION_DATE = <char *> BLOSC_VERSION_DATE
VERSION_STRING = VERSION_STRING.decode()
VERSION_DATE = VERSION_DATE.decode()
__version__ = VERSION_STRING
NOSHUFFLE = BLOSC_NOSHUFFLE
SHUFFLE = BLOSC_SHUFFLE
BITSHUFFLE = BLOSC_BITSHUFFLE
# automatic shuffle
AUTOSHUFFLE = -1
# automatic block size - let blosc decide
AUTOBLOCKS = 0

# synchronization
try:
    mutex = multiprocessing.Lock()
except OSError:
    mutex = None
except ImportError:
    mutex = None

# store ID of process that first loads the module, so we can detect a fork later
_importer_pid = os.getpid()


def init():
    """Initialize the Blosc library environment."""
    blosc_init()


def destroy():
    """Destroy the Blosc library environment."""
    blosc_destroy()


def compname_to_compcode(cname):
    """Return the compressor code associated with the compressor name. If the compressor
    name is not recognized, or there is not support for it in this build, -1 is returned
    instead."""
    if isinstance(cname, str):
        cname = cname.encode('ascii')
    return blosc_compname_to_compcode(cname)


def list_compressors():
    """Get a list of compressors supported in the current build."""
    s = blosc_list_compressors()
    s = s.decode('ascii')
    return s.split(',')


def get_nthreads():
    """Get the number of threads that Blosc uses internally for compression and
    decompression."""
    return blosc_get_nthreads()


def set_nthreads(int nthreads):
    """Set the number of threads that Blosc uses internally for compression and
    decompression."""
    return blosc_set_nthreads(nthreads)


def cbuffer_sizes(source):
    """Return information about a compressed buffer, namely the number of uncompressed
    bytes (`nbytes`) and compressed (`cbytes`).  It also returns the `blocksize` (which
    is used internally for doing the compression by blocks).

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


def cbuffer_complib(source):
    """Return the name of the compression library used to compress `source`."""
    cdef:
        Buffer buffer

    # obtain buffer
    buffer = Buffer(source, PyBUF_ANY_CONTIGUOUS)

    # determine buffer size
    complib = blosc_cbuffer_complib(buffer.ptr)

    # release buffers
    buffer.release()

    complib = complib.decode('ascii')

    return complib


def cbuffer_metainfo(source):
    """Return some meta-information about the compressed buffer in `source`, including
    the typesize, whether the shuffle or bit-shuffle filters were used, and the
    whether the buffer was memcpyed.

    Returns
    -------
    typesize
    shuffle
    memcpyed

    """
    cdef:
        Buffer buffer
        size_t typesize
        int flags

    # obtain buffer
    buffer = Buffer(source, PyBUF_ANY_CONTIGUOUS)

    # determine buffer size
    blosc_cbuffer_metainfo(buffer.ptr, &typesize, &flags)

    # release buffers
    buffer.release()

    # decompose flags
    if flags & BLOSC_DOSHUFFLE:
        shuffle = SHUFFLE
    elif flags & BLOSC_DOBITSHUFFLE:
        shuffle = BITSHUFFLE
    else:
        shuffle = NOSHUFFLE
    memcpyed = flags & BLOSC_MEMCPYED

    return typesize, shuffle, memcpyed


def err_bad_cname(cname):
    raise ValueError('bad compressor or compressor not supported: %r; expected one of '
                     '%s' % (cname, list_compressors()))


def compress(source, char* cname, int clevel, int shuffle=SHUFFLE,
             int blocksize=AUTOBLOCKS):
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
        Either NOSHUFFLE (0), SHUFFLE (1), BITSHUFFLE (2) or AUTOSHUFFLE (-1). If -1
        (default), bit-shuffle will be used for buffers with itemsize 1,
        and byte-shuffle will be used otherwise.
    blocksize : int
        The requested size of the compressed blocks.  If 0, an automatic blocksize will
        be used.

    Returns
    -------
    dest : bytes
        Compressed data.

    """

    cdef:
        char *source_ptr
        char *dest_ptr
        Buffer source_buffer
        size_t nbytes, itemsize
        int cbytes
        bytes dest

    # check valid cname early
    cname_str = cname.decode('ascii')
    if cname_str not in list_compressors():
        err_bad_cname(cname_str)

    # setup source buffer
    source_buffer = Buffer(source, PyBUF_ANY_CONTIGUOUS)
    source_ptr = source_buffer.ptr
    nbytes = source_buffer.nbytes
    itemsize = source_buffer.itemsize

    # determine shuffle
    if shuffle == AUTOSHUFFLE:
        if itemsize == 1:
            shuffle = BITSHUFFLE
        else:
            shuffle = SHUFFLE
    elif shuffle not in [NOSHUFFLE, SHUFFLE, BITSHUFFLE]:
        raise ValueError('invalid shuffle argument; expected -1, 0, 1 or 2, found %r' %
                         shuffle)

    try:

        # setup destination
        dest = PyBytes_FromStringAndSize(NULL, nbytes + BLOSC_MAX_OVERHEAD)
        dest_ptr = PyBytes_AS_STRING(dest)

        # perform compression
        if _get_use_threads():
            # allow blosc to use threads internally

            # N.B., we are using blosc's global context, and so we need to use a lock
            # to ensure no-one else can modify the global context while we're setting it
            # up and using it.
            with mutex:

                # set compressor
                compressor_set = blosc_set_compressor(cname)
                if compressor_set < 0:
                    # shouldn't happen if we checked against list of compressors
                    # already, but just in case
                    err_bad_cname(cname_str)

                # set blocksize
                blosc_set_blocksize(blocksize)

                # perform compression
                with nogil:
                    cbytes = blosc_compress(clevel, shuffle, itemsize, nbytes, source_ptr,
                                            dest_ptr, nbytes + BLOSC_MAX_OVERHEAD)

        else:
            with nogil:
                cbytes = blosc_compress_ctx(clevel, shuffle, itemsize, nbytes, source_ptr,
                                            dest_ptr, nbytes + BLOSC_MAX_OVERHEAD,
                                            cname, blocksize, 1)

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
        Compressed data, including blosc header. Can be any object supporting the buffer
        protocol.
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
        arr = ensure_contiguous_ndarray(dest)
        dest_buffer = Buffer(arr, PyBUF_ANY_CONTIGUOUS | PyBUF_WRITEABLE)
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


def decompress_partial(source, start, nitems, dest=None):
    """**Experimental**
    Decompress data of only a part of a buffer.

    Parameters
    ----------
    source : bytes-like
        Compressed data, including blosc header. Can be any object supporting the buffer
        protocol.
    start: int,
        Offset in item where we want to start decoding
    nitems: int
        Number of items we want to decode
    dest : array-like, optional
        Object to decompress into.


    Returns
    -------
    dest : bytes
        Object containing decompressed data.

    """
    cdef:
        int ret
        int encoding_size
        int nitems_bytes 
        int start_bytes
        char *source_ptr
        char *dest_ptr
        Buffer source_buffer
        Buffer dest_buffer = None

    # setup source buffer
    source_buffer = Buffer(source, PyBUF_ANY_CONTIGUOUS)
    source_ptr = source_buffer.ptr

    # get encoding size from source buffer header
    encoding_size = source[3]

    # convert varibles to handle type and encoding sizes
    nitems_bytes = nitems * encoding_size
    start_bytes = (start * encoding_size)

    # setup destination buffer
    if dest is None:
        dest = PyBytes_FromStringAndSize(NULL, nitems_bytes)
        dest_ptr = PyBytes_AS_STRING(dest)
        dest_nbytes = nitems_bytes
    else:
        arr = ensure_contiguous_ndarray(dest)
        dest_buffer = Buffer(arr, PyBUF_ANY_CONTIGUOUS | PyBUF_WRITEABLE)
        dest_ptr = dest_buffer.ptr
        dest_nbytes = dest_buffer.nbytes

    # try decompression
    try:
        if dest_nbytes < nitems_bytes:
            raise ValueError('destination buffer too small; expected at least %s, '
                             'got %s' % (nitems_bytes, dest_nbytes))
        ret = blosc_getitem(source_ptr, start, nitems, dest_ptr)

    finally:
        source_buffer.release()
        if dest_buffer is not None:
            dest_buffer.release()

    # ret refers to the number of bytes returned from blosc_getitem. 
    if ret <= 0:
        raise RuntimeError('error during blosc partial decompression: %d', ret)

    return dest
        

# set the value of this variable to True or False to override the
# default adaptive behaviour
use_threads = None


def _get_use_threads():
    global use_threads
    proc = multiprocessing.current_process()

    # check if locks are available, and if not no threads
    if not mutex:
        return False

    # check for fork
    if proc.pid != _importer_pid:
        # If this module has been imported in the parent process, and the current process
        # is a fork, attempting to use blosc in multi-threaded mode will cause a
        # program hang, so we force use of blosc ctx functions, i.e., no threads.
        return False

    if use_threads in [True, False]:
        # user has manually overridden the default behaviour
        _use_threads = use_threads

    else:
        # Adaptive behaviour: allow blosc to use threads if it is being called from the
        # main Python thread in the main Python process, inferring that it is being run
        # from within a single-threaded, single-process program; otherwise do not allow
        # blosc to use threads, inferring it is being run from within a multi-threaded
        # program or multi-process program

        if proc.name != 'MainProcess':
            _use_threads = False
        elif hasattr(threading, 'main_thread'):
            _use_threads = (threading.main_thread() == threading.current_thread())
        else:
            _use_threads = threading.current_thread().name == 'MainThread'

    return _use_threads


_shuffle_repr = ['AUTOSHUFFLE', 'NOSHUFFLE', 'SHUFFLE', 'BITSHUFFLE']


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
        Either NOSHUFFLE (0), SHUFFLE (1), BITSHUFFLE (2) or AUTOSHUFFLE (-1). If -1
        (default), bit-shuffle will be used for buffers with itemsize 1,
        and byte-shuffle will be used otherwise.
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
        buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
        return compress(buf, self._cname_bytes, self.clevel, self.shuffle, self.blocksize)

    def decode(self, buf, out=None):
        buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
        return decompress(buf, out)

    def decode_partial(self, buf, int start, int nitems, out=None):
        '''**Experimental**'''
        buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
        return decompress_partial(buf, start, nitems, dest=out)

    def __repr__(self):
        r = '%s(cname=%r, clevel=%r, shuffle=%s, blocksize=%s)' % \
            (type(self).__name__,
             self.cname,
             self.clevel,
             _shuffle_repr[self.shuffle + 1],
             self.blocksize)
        return r
