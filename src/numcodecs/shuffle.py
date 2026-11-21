from typing import cast, overload

import numpy as np
import numpy.typing as npt

from ._shuffle import _doShuffle, _doUnshuffle
from .abc import Codec, CodecConfig
from .compat import ensure_contiguous_ndarray
from .ndarray_like import BufferLike


class ShuffleConfig(CodecConfig):
    """The configuration of a `Shuffle` codec, as returned by `Shuffle.get_config`."""

    elementsize: int


class Shuffle(Codec):
    """Codec providing shuffle

    Parameters
    ----------
    elementsize : int
        Size in bytes of the array elements.  Default = 4

    """

    codec_id = 'shuffle'

    def __init__(self, elementsize: int = 4) -> None:
        self.elementsize = elementsize

    def get_config(self) -> ShuffleConfig:
        return cast(ShuffleConfig, super().get_config())

    def _prepare_arrays(
        self, buf: BufferLike, out: BufferLike | None
    ) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.generic]]:
        buf = ensure_contiguous_ndarray(buf)

        if out is None:
            out = np.zeros(buf.nbytes, dtype='uint8')
        else:
            out = ensure_contiguous_ndarray(out)

        if self.elementsize <= 1:
            out.view(buf.dtype)[: len(buf)] = buf[:]  # no shuffling needed
            return buf, out

        if buf.nbytes % self.elementsize != 0:
            raise ValueError("Shuffle buffer is not an integer multiple of elementsize")

        return buf, out

    @overload
    def encode(self, buf: BufferLike, out: None = None) -> npt.NDArray[np.uint8]: ...
    @overload
    def encode[OutT: BufferLike](self, buf: BufferLike, out: OutT) -> OutT: ...
    def encode(self, buf: BufferLike, out: BufferLike | None = None) -> BufferLike:
        buf, out_arr = self._prepare_arrays(buf, out)

        if self.elementsize > 1:
            _doShuffle(buf.view("uint8"), out_arr.view("uint8"), self.elementsize)

        return out_arr if out is None else out

    @overload
    def decode(self, buf: BufferLike, out: None = None) -> npt.NDArray[np.uint8]: ...
    @overload
    def decode[OutT: BufferLike](self, buf: BufferLike, out: OutT) -> OutT: ...
    def decode(self, buf: BufferLike, out: BufferLike | None = None) -> BufferLike:
        buf, out_arr = self._prepare_arrays(buf, out)

        if self.elementsize > 1:
            _doUnshuffle(buf.view("uint8"), out_arr.view("uint8"), self.elementsize)

        return out_arr if out is None else out

    def __repr__(self) -> str:
        return f'{type(self).__name__}(elementsize={self.elementsize})'
