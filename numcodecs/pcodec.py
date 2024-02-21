from typing import Optional, Literal

import numcodecs
import numcodecs.abc
from numcodecs.compat import ensure_contiguous_ndarray

try:
    import pcodec
except ImportError:
    pcodec = None


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
    param max_page_n : int
        The maximum number of values to encoder per pcodec page.
        If set too high or too low, pcodec's compression ratio may drop.
    """

    codec_id = "pcodec"

    def __init__(
        self,
        level: int = 8,
        delta_encoding_order: Optional[int] = None,
        int_mult_spec: Literal["enabled", "disabled"] = "enabled",
        float_mult_spec: Literal["enabled", "disabled"] = "enabled",
        max_page_n: int = 262144,
    ):
        if pcodec is None:
            raise ImportError("pcodec is not available. Please install the pcodec package.")

        # note that we use `level` instead of `compression_level` to
        # match other codecs
        self.level = level
        self.delta_encoding_order = delta_encoding_order
        self.int_mult_spec = int_mult_spec
        self.float_mult_spec = float_mult_spec
        self.max_page_n = max_page_n

    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf)
        return pcodec.auto_compress(
            buf,
            compression_level=self.level,
            delta_encoding_order=self.delta_encoding_order,
            int_mult_spec=self.int_mult_spec,
            float_mult_spec=self.float_mult_spec,
            max_page_n=self.max_page_n,
        )

    def decode(self, buf, out=None):
        if out is not None:
            out = ensure_contiguous_ndarray(out)
            pcodec.simple_decompress_into(buf, out)
            return out
        else:
            return pcodec.auto_decompress(buf)
