import numpy as np

from .compat import ensure_ndarray_like
from .bitround import BitRound

# The size in bits of the mantissa/significand for the various floating types
# You cannot keep more bits of data than you have available
# https://en.wikipedia.org/wiki/IEEE_754

NMBITS = {64: 12, 32: 9, 16: 6}  # number of non mantissa bits for given dtype


class BitInfo(BitRound):
    """Floating-point bit information codec

    Drops bits from the floating point mantissa, leaving an array more amenable
    to compression. The number of bits to keep is determined using the approach
    from Klöwer et al. 2021 (https://www.nature.com/articles/s43588-021-00156-2).
    See https://github.com/zarr-developers/numcodecs/issues/298 for discussion
    and the original implementation in Julia referred to at
    https://github.com/milankl/BitInformation.jl

    Parameters
    ----------

    info_level: float
        The level of information to preserve in the data. The value should be
        between 0. and 1.0. Higher values preserve more information.

    axes: int or list of int, optional
        Axes along which to calculate the bit information. If None, all axes
        are used.
    """

    codec_id = 'bitinfo'

    def __init__(self, info_level: float, axes=None):
        if (info_level < 0) or (info_level > 1.0):
            raise ValueError("Please provide `info_level` from interval [0.,1.]")

        elif axes is not None and not isinstance(axes, list):
            if int(axes) != axes:
                raise ValueError("axis must be an integer or a list of integers.")
            axes = [axes]

        elif isinstance(axes, list) and not all(int(ax) == ax for ax in axes):
            raise ValueError("axis must be an integer or a list of integers.")

        self.info_level = info_level
        self.axes = axes

    def encode(self, buf):
        """Create int array by rounding floating-point data

        The itemsize will be preserved, but the output should be much more
        compressible.
        """
        a = ensure_ndarray_like(buf)
        dtype = a.dtype

        if not a.dtype.kind == "f" or a.dtype.itemsize > 8:
            raise TypeError("Only float arrays (16-64bit) can be bit-rounded")

        if self.axes is None:
            self.axes = range(a.ndim)

        itemsize = a.dtype.itemsize
        astype = f"u{itemsize}"
        if a.dtype in (np.float16, np.float32, np.float64):
            a = signed_exponent(a)

        a = a.astype(astype)
        keepbits = []

        for ax in self.axes:
            info_per_bit = bitinformation(a, axis=ax)
            keepbits.append(get_keepbits(info_per_bit, self.info_level))

        keepbits = max(keepbits)

        return BitRound.bitround(buf, keepbits, dtype)


def exponent_bias(dtype):
    """
    Returns the exponent bias for a given floating-point dtype.

    Example
    -------
    >>> exponent_bias("f4")
    127
    >>> exponent_bias("f8")
    1023
    """
    info = np.finfo(dtype)
    exponent_bits = info.bits - info.nmant - 1
    return 2 ** (exponent_bits - 1) - 1


def exponent_mask(dtype):
    """
    Returns exponent mask for a given floating-point dtype.

    Example
    -------
    >>> np.binary_repr(exponent_mask(np.float32), width=32)
    '01111111100000000000000000000000'
    >>> np.binary_repr(exponent_mask(np.float16), width=16)
    '0111110000000000'
    """
    if dtype == np.float16:
        mask = 0x7C00
    elif dtype == np.float32:
        mask = 0x7F80_0000
    elif dtype == np.float64:
        mask = 0x7FF0_0000_0000_0000
    else:
        raise ValueError(f"Unsupported dtype {dtype}")
    return mask


def signed_exponent(A):
    """
    Transform biased exponent notation to signed exponent notation.

    Parameters
    ----------
    a : array
        Array to transform

    Returns
    -------
    array

    Example
    -------
    >>> A = np.array(0.03125, dtype="float32")
    >>> np.binary_repr(A.view("uint32"), width=32)
    '00111101000000000000000000000000'
    >>> np.binary_repr(signed_exponent(A), width=32)
    '01000010100000000000000000000000'
    >>> A = np.array(0.03125, dtype="float64")
    >>> np.binary_repr(A.view("uint64"), width=64)
    '0011111110100000000000000000000000000000000000000000000000000000'
    >>> np.binary_repr(signed_exponent(A), width=64)
    '0100000001010000000000000000000000000000000000000000000000000000'
    """
    itemsize = A.dtype.itemsize
    uinttype = f"u{itemsize}"
    inttype = f"i{itemsize}"

    sign_mask = 1 << np.finfo(A.dtype).bits - 1
    sfmask = sign_mask | (1 << np.finfo(A.dtype).nmant) - 1
    emask = exponent_mask(A.dtype)
    esignmask = sign_mask >> 1

    sbits = np.finfo(A.dtype).nmant
    if itemsize == 8:
        sbits = np.uint64(sbits)
        emask = np.uint64(emask)
    bias = exponent_bias(A.dtype)

    ui = A.view(uinttype)
    sf = ui & sfmask
    e = ((ui & emask) >> sbits).astype(inttype) - bias
    max_eabs = np.iinfo(A.view(uinttype).dtype).max >> sbits
    eabs = abs(e) % (max_eabs + 1)
    esign = np.where(e < 0, esignmask, 0)
    if itemsize == 8:
        eabs = np.uint64(eabs)
        esign = np.uint64(esign)
    esigned = esign | (eabs << sbits)
    return (sf | esigned).view(np.int64)


def bitpaircount_u1(a, b):
    assert a.dtype == "u1"
    assert b.dtype == "u1"

    unpack_a = np.unpackbits(a.flatten()).astype("u1")
    unpack_b = np.unpackbits(b.flatten()).astype("u1")

    index = ((unpack_a << 1) | unpack_b).reshape(-1, 8)

    selection = np.array([0, 1, 2, 3], dtype="u1")
    sel = np.where((index[..., np.newaxis]) == selection, True, False)
    return sel.sum(axis=0).reshape([8, 2, 2])


def bitpaircount(a, b):
    assert a.dtype.kind == "u"
    assert b.dtype.kind == "u"

    nbytes = max(a.dtype.itemsize, b.dtype.itemsize)

    a, b = np.broadcast_arrays(a, b)

    bytewise_counts = []
    for i in range(nbytes):
        s = (nbytes - 1 - i) * 8
        bitc = bitpaircount_u1((a >> s).astype("u1"), (b >> s).astype("u1"))
        bytewise_counts.append(bitc)
    return np.concatenate(bytewise_counts, axis=0)


def mutual_information(a, b, base=2):
    """Calculate the mutual information between two arrays.
    """
    assert a.dtype == b.dtype
    assert a.dtype.kind == "u"

    size = np.prod(np.broadcast_shapes(a.shape, b.shape))
    counts = bitpaircount(a, b)

    p = counts.astype("float") / size
    p = np.ma.masked_equal(p, 0)
    pr = p.sum(axis=-1)[..., np.newaxis]
    ps = p.sum(axis=-2)[..., np.newaxis, :]
    mutual_info = (p * np.ma.log(p / (pr * ps))).sum(axis=(-1, -2)) / np.log(base)
    return mutual_info


def bitinformation(a, axis=0):
    """Get the information content of each bit in the array.

    Parameters
    ----------
    a : array
        Array to calculate the bit information.
    axis : int
        Axis along which to calculate the bit information.

    Returns
    -------
    info_per_bit : array
    """
    assert a.dtype.kind == "u"

    sa = tuple(slice(0, -1) if i == axis else slice(None) for i in range(len(a.shape)))
    sb = tuple(
        slice(1, None) if i == axis else slice(None) for i in range(len(a.shape))
    )
    return mutual_information(a[sa], a[sb])


def get_keepbits(info_per_bit, inflevel=0.99):
    """Get the number of mantissa bits to keep.

    Parameters
    ----------
    info_per_bit : array
      Information content of each bit from `get_bitinformation`.

    inflevel : float
      Level of information that shall be preserved.

    Returns
    -------
    keepbits : int
      Number of mantissa bits to keep

    """
    if (inflevel < 0) or (inflevel > 1.0):
        raise ValueError("Please provide `inflevel` from interval [0.,1.]")

    cdf = _cdf_from_info_per_bit(info_per_bit)
    bitdim_non_mantissa_bits = NMBITS[len(info_per_bit)]
    keepmantissabits = (
        (cdf > inflevel).argmax() + 1 - bitdim_non_mantissa_bits
    )

    return keepmantissabits


def _cdf_from_info_per_bit(info_per_bit):
    """Convert info_per_bit to cumulative distribution function"""
    tol = info_per_bit[-4:].max() * 1.5
    info_per_bit[info_per_bit < tol] = 0
    cdf = info_per_bit.cumsum()
    return cdf / cdf[-1]
