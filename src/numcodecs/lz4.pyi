from typing import Final, overload

from .abc import Codec, CodecConfig
from .ndarray_like import BufferLike

VERSION_STRING: Final[str]
DEFAULT_ACCELERATION: Final[int]

def compress(source: BufferLike, acceleration: int = ...) -> bytes: ...
@overload
def decompress(source: BufferLike, dest: None = None) -> bytes: ...
@overload
def decompress[OutT: BufferLike](source: BufferLike, dest: OutT) -> OutT: ...

class LZ4Config(CodecConfig):
    acceleration: int

class LZ4(Codec):
    codec_id: str
    max_buffer_size: int
    acceleration: int
    def __init__(self, acceleration: int = ...) -> None: ...
    def get_config(self) -> LZ4Config: ...
    def encode(self, buf: BufferLike) -> bytes: ...
    @overload
    def decode(self, buf: BufferLike, out: None = None) -> bytes: ...
    @overload
    def decode[OutT: BufferLike](self, buf: BufferLike, out: OutT) -> OutT: ...
