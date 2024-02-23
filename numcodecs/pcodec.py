from typing import Optional, Literal

import numcodecs
import numcodecs.abc
from numcodecs.compat import ensure_contiguous_ndarray

try:
    from pcodec import standalone, ChunkConfig, PagingSpec
except ImportError:
    standalone = None


DEFAULT_MAX_PAGE_N = 262144


class PCodec(numcodecs.abc.Codec):
    """
    PCodec (or pco, pronounced "pico") losslessly compresses and decompresses
    numerical sequences with high compression ratio and fast speed.

    See `PCodec Repo <https://github.com/mwlon/pcodec>`_ for more information.

    PCodec supports only the following numerical dtypes: uint32, unit64, int32,
    int64, float32, and float64.

    Parameters
    ----------
    level : int
        A compression level from 0-12, where 12 take the longest and compresses
        the most.
    delta_encoding_order : init or None
        Either a delta encoding level from 0-7 or None. If set to None, pcodec
        will try to infer the optimal delta encoding order.
    int_mult_spec : {'enabled', 'disabled'}
        If enabled, pcodec will consider using int mult mode, which can
        substantially improve compression ratio but decrease speed in some cases
        for integer types.
    float_mult_spec : {'enabled', 'disabled'}
        If enabled, pcodec will consider using float mult mode, which can
        substantially improve compression ratio but decrease speed in some cases
        for float types.
    equal_pages_up_to : int
        Divide the chunk into equal pages of up to this many numbers.
    """

    codec_id = "pcodec"

    def __init__(
        self,
        level: int = 8,
        delta_encoding_order: Optional[int] = None,
        int_mult_spec: Literal["enabled", "disabled"] = "enabled",
        float_mult_spec: Literal["enabled", "disabled"] = "enabled",
        equal_pages_up_to: int = 262144
    ):
        if standalone is None:
            raise ImportError(
                "pcodec must be installed to use the PCodec codec."
            )

        # note that we use `level` instead of `compression_level` to
        # match other codecs
        self.level = level
        self.delta_encoding_order = delta_encoding_order
        self.int_mult_spec = int_mult_spec
        self.float_mult_spec = float_mult_spec
        self.equal_pages_up_to = equal_pages_up_to

    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf)

        paging_spec = PagingSpec.equal_pages_up_to(self.equal_pages_up_to)

        config = ChunkConfig(
            compression_level=self.level,
            delta_encoding_order=self.delta_encoding_order,
            int_mult_spec=self.int_mult_spec,
            float_mult_spec=self.float_mult_spec,
            paging_spec=paging_spec,
        )
        return standalone.simple_compress(buf, config)

    def decode(self, buf, out=None):
        if out is not None:
            out = ensure_contiguous_ndarray(out)
            standalone.simple_decompress_into(buf, out)
            return out
        else:
            return standalone.simple_decompress(buf)
