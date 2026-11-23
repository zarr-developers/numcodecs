from typing import Final, overload

from .abc import Codec, CodecConfig
from .ndarray_like import BufferLike

VERSION_NUMBER: Final[int]
MAJOR_VERSION_NUMBER: Final[int]
MINOR_VERSION_NUMBER: Final[int]
MICRO_VERSION_NUMBER: Final[int]
DEFAULT_CLEVEL: Final[int]
MAX_CLEVEL: Final[int]

def compress(source: BufferLike, level: int = ..., checksum: bool = False) -> bytes: ...
@overload
def decompress(source: BufferLike, dest: None = None) -> bytes: ...
@overload
def decompress[OutT: BufferLike](source: BufferLike, dest: OutT) -> OutT: ...

class ZstdConfig(CodecConfig):
    level: int
    checksum: bool

class Zstd(Codec):
    codec_id: str
    max_buffer_size: int
    level: int
    checksum: bool
    def __init__(self, level: int = ..., checksum: bool = False) -> None: ...
    def get_config(self) -> ZstdConfig: ...
    def encode(self, buf: BufferLike) -> bytes: ...
    @overload
    def decode(self, buf: BufferLike, out: None = None) -> bytes: ...
    @overload
    def decode[OutT: BufferLike](self, buf: BufferLike, out: OutT) -> OutT: ...
    @classmethod
    def default_level(cls) -> int: ...
    @classmethod
    def min_level(cls) -> int: ...
    @classmethod
    def max_level(cls) -> int: ...
