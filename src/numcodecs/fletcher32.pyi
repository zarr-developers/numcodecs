from typing import overload

import numpy as np
import numpy.typing as npt

from .abc import Codec, CodecConfig
from .ndarray_like import BufferLike

class Fletcher32(Codec):
    codec_id: str
    def get_config(self) -> CodecConfig: ...
    def encode(self, buf: BufferLike) -> bytes: ...
    @overload
    def decode(self, buf: BufferLike, out: None = None) -> npt.NDArray[np.uint8]: ...
    @overload
    def decode[OutT: BufferLike](self, buf: BufferLike, out: OutT) -> OutT: ...
