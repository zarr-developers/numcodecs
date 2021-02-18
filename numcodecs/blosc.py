# native python wrapper around blosc
# cython: embedsignature=True
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# cython: language_level=3
import threading
import multiprocessing
import os
import blosc


from .compat import ensure_contiguous_ndarray
from .abc import Codec


# automatic shuffle
AUTOSHUFFLE = -1
# automatic block size - let blosc decide
AUTOBLOCKS = 0

# synchronization
try:
    mutex = multiprocessing.Lock()
except OSError:
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


def err_bad_cname(cname):
    raise ValueError('bad compressor or compressor not supported: %r; expected one of '
                     '%s' % (cname, list_compressors()))


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
    NOSHUFFLE = 0
    SHUFFLE = 1
    BITSHUFFLE = 2
    AUTOSHUFFLE = -1
    max_buffer_size = 2**31 - 1

    def __init__(self, cname='lz4', clevel=5, shuffle=SHUFFLE, blocksize=AUTOBLOCKS):
        self.cname = cname
        if isinstance(cname, str):
            self._cname_bytes = cname.encode('ascii')
        else:
            self._cname_bytes = cname
        self.clevel = clevel
        self.shuffle = shuffle
        if shuffle == Blosc.AUTOSHUFFLE:
            # FIXME: where to find itemsize?
            self.shuffle = Blosc.NOSHUFFLE
        self.blocksize = blocksize

    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
        return blosc.compress(
            buf,
            # typesize=self.blocksize, FIXME
            clevel=self.clevel,
            shuffle=self.shuffle,
            cname=self.cname,
        )

    def decode(self, buf): # FIXME , out=None):
        buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
        return blosc.decompress(buf)

    def decode_partial(self, buf, start, nitems, out=None):
        '''**Experimental**'''
        buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
        # return blosc.decompress_partial(buf, start, nitems, dest=out)
        raise Exception("FIXME")


    def __repr__(self):
        r = '%s(cname=%r, clevel=%r, shuffle=%s, blocksize=%s)' % \
            (type(self).__name__,
             self.cname,
             self.clevel,
             _shuffle_repr[self.shuffle + 1],
             self.blocksize)
        return r

def compress(source, cname, clevel: int, shuffle:int = SHUFFLE,
             blocksize:int = AUTOBLOCKS):
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
