from collections.abc import Iterable
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from .abc import Codec, CodecConfig
from .compat import ensure_ndarray, ensure_text, ndarray_copy
from .ndarray_like import BufferLike


class CategorizeConfig(CodecConfig):
    """The configuration of a `Categorize` codec, as returned by `Categorize.get_config`."""

    labels: list[str]
    dtype: str
    astype: str


class Categorize(Codec):
    """Filter encoding categorical string data as integers.

    Parameters
    ----------
    labels : sequence of strings
        Category labels.
    dtype : dtype
        Data type to use for decoded data.
    astype : dtype, optional
        Data type to use for encoded data.

    Examples
    --------
    >>> import numcodecs
    >>> import numpy as np
    >>> x = np.array(['male', 'female', 'female', 'male', 'unexpected'], dtype=object)
    >>> x
    array(['male', 'female', 'female', 'male', 'unexpected'],
          dtype=object)
    >>> codec = numcodecs.Categorize(labels=['female', 'male'], dtype=object)
    >>> y = codec.encode(x)
    >>> y
    array([2, 1, 1, 2, 0], dtype=uint8)
    >>> z = codec.decode(y)
    >>> z
    array(['male', 'female', 'female', 'male', ''],
          dtype=object)

    """

    codec_id = 'categorize'

    def __init__(
        self,
        labels: Iterable[str | BufferLike],
        dtype: npt.DTypeLike,
        astype: npt.DTypeLike = 'u1',
    ) -> None:
        self.dtype: np.dtype[Any] = np.dtype(dtype)
        if self.dtype.kind not in 'UO':
            raise TypeError("only unicode ('U') and object ('O') dtypes are supported")
        self.labels: list[str] = [ensure_text(label) for label in labels]
        self.astype: np.dtype[Any] = np.dtype(astype)
        if self.astype == np.dtype(object):
            raise TypeError('encoding as object array not supported')

    def encode(self, buf: BufferLike | Iterable[str]) -> npt.NDArray[Any]:
        # normalise input
        if self.dtype == np.dtype(object):
            arr = np.asarray(buf, dtype=object)
        else:
            # only an object dtype accepts a plain iterable; anything else must be a buffer
            arr = ensure_ndarray(buf).view(self.dtype)  # type: ignore[arg-type]

        # flatten to simplify implementation
        arr = arr.reshape(-1, order='A')

        # setup output array
        enc = np.zeros_like(arr, dtype=self.astype)

        # apply encoding, reserving 0 for values not specified in labels
        for i, label in enumerate(self.labels):
            enc[arr == label] = i + 1

        return enc

    @overload
    def decode(self, buf: BufferLike, out: None = None) -> npt.NDArray[Any]: ...
    @overload
    def decode[OutT: BufferLike](self, buf: BufferLike, out: OutT) -> OutT: ...
    def decode(self, buf: BufferLike, out: BufferLike | None = None) -> BufferLike:
        # normalise input
        enc = ensure_ndarray(buf).view(self.astype)

        # flatten to simplify implementation
        enc = enc.reshape(-1, order='A')

        # setup output
        dec = np.full_like(enc, fill_value='', dtype=self.dtype)

        # apply decoding
        for i, label in enumerate(self.labels):
            dec[enc == (i + 1)] = label

        # handle output
        return ndarray_copy(dec, out)

    def get_config(self) -> CategorizeConfig:
        return {
            'id': self.codec_id,
            'labels': self.labels,
            'dtype': self.dtype.str,
            'astype': self.astype.str,
        }

    def __repr__(self) -> str:
        # make sure labels part is not too long
        labels = repr(self.labels[:3])
        if len(self.labels) > 3:
            labels = labels[:-1] + ', ...]'
        return f'{type(self).__name__}(dtype={self.dtype.str!r}, astype={self.astype.str!r}, labels={labels})'
