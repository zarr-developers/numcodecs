from typing import Final, overload

from .abc import Codec, CodecConfig
from .ndarray_like import BufferLike

NOSHUFFLE: Final[int]
SHUFFLE: Final[int]
BITSHUFFLE: Final[int]
AUTOSHUFFLE: Final[int]
AUTOBLOCKS: Final[int]
MAX_OVERHEAD: Final[int]
MAX_BUFFERSIZE: Final[int]
MAX_THREADS: Final[int]
MAX_TYPESIZE: Final[int]
VERSION_STRING: Final[str]
VERSION_DATE: Final[str]
__version__: Final[str]

use_threads: bool | None
"""Set to `True` or `False` to override whether Blosc may use threads; `None` (the
default) decides from the calling thread and process."""

def _init() -> None: ...
def _destroy() -> None: ...
def _get_use_threads() -> bool: ...
def _cbuffer_sizes(source: BufferLike) -> tuple[int, int, int]: ...
def _cbuffer_metainfo(source: BufferLike) -> tuple[int, bool, bool]: ...
def list_compressors() -> list[str]: ...
def get_nthreads() -> int: ...
def set_nthreads(nthreads: int) -> int: ...
def cbuffer_complib(source: BufferLike) -> str: ...
def compress(
    source: BufferLike,
    cname: bytes,
    clevel: int,
    shuffle: int = ...,
    blocksize: int = ...,
    typesize: int | None = None,
) -> bytes: ...
@overload
def decompress(source: BufferLike, dest: None = None) -> bytes: ...
@overload
def decompress[OutT: BufferLike](source: BufferLike, dest: OutT) -> OutT: ...

class BloscConfig(CodecConfig):
    cname: str
    clevel: int
    shuffle: int
    blocksize: int

class Blosc(Codec):
    codec_id: str
    NOSHUFFLE: int
    SHUFFLE: int
    BITSHUFFLE: int
    AUTOSHUFFLE: int
    max_buffer_size: int
    cname: str
    clevel: int
    shuffle: int
    blocksize: int
    def __init__(
        self,
        cname: str = "lz4",
        clevel: int = 5,
        shuffle: int = ...,
        blocksize: int = ...,
        typesize: int | None = None,
    ) -> None: ...
    def get_config(self) -> BloscConfig: ...
    def encode(self, buf: BufferLike) -> bytes: ...
    @overload
    def decode(self, buf: BufferLike, out: None = None) -> bytes: ...
    @overload
    def decode[OutT: BufferLike](self, buf: BufferLike, out: OutT) -> OutT: ...
