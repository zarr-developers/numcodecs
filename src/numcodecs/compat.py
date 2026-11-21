import array
import codecs
from collections.abc import Buffer
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from .ndarray_like import BufferLike, NDArrayLike, is_ndarray_like


@overload
def ensure_ndarray_like(buf: Buffer) -> npt.NDArray[Any]: ...
@overload
def ensure_ndarray_like(buf: NDArrayLike) -> NDArrayLike: ...
def ensure_ndarray_like(buf: BufferLike) -> NDArrayLike:
    """Convenience function to coerce `buf` to ndarray-like array.

    Parameters
    ----------
    buf : ndarray-like, array-like, or bytes-like
        A numpy array like object such as numpy.ndarray, cupy.ndarray, or
        any object exporting a buffer interface.

    Returns
    -------
    arr : NDArrayLike
        A ndarray-like, sharing memory with `buf`.

    Notes
    -----
    This function will not create a copy under any circumstances, it is guaranteed to
    return a view on memory exported by `buf`.
    """

    if not is_ndarray_like(buf):
        if isinstance(buf, array.array) and buf.typecode in "cu":
            # Guard condition, do not support array.array with unicode type, this is
            # problematic because numpy does not support it on all platforms. Also do not
            # support char as it was removed in Python 3.
            raise TypeError("array.array with char or unicode type is not supported")
        else:
            # N.B., first take a memoryview to make sure that we subsequently create a
            # numpy array from a memory buffer with no copy
            mem = memoryview(buf)
            # instantiate array from memoryview, ensures no copy
            buf = np.array(mem, copy=False)
    return buf


def ensure_ndarray(buf: BufferLike) -> npt.NDArray[Any]:
    """Convenience function to coerce `buf` to a numpy array, if it is not already a
    numpy array.

    Parameters
    ----------
    buf : array-like or bytes-like
        A numpy array or any object exporting a buffer interface.

    Returns
    -------
    arr : ndarray
        A numpy array, sharing memory with `buf`.

    Notes
    -----
    This function will not create a copy under any circumstances, it is guaranteed to
    return a view on memory exported by `buf`.
    """
    return np.array(ensure_ndarray_like(buf), copy=False)


@overload
def ensure_contiguous_ndarray_like(
    buf: Buffer, max_buffer_size: int | None = None, flatten: bool = True
) -> npt.NDArray[Any]: ...
@overload
def ensure_contiguous_ndarray_like(
    buf: NDArrayLike, max_buffer_size: int | None = None, flatten: bool = True
) -> NDArrayLike: ...
def ensure_contiguous_ndarray_like(
    buf: BufferLike, max_buffer_size: int | None = None, flatten: bool = True
) -> NDArrayLike:
    """Convenience function to coerce `buf` to ndarray-like array.
    Also ensures that the returned value exports fully contiguous memory,
    and supports the new-style buffer interface. If the optional max_buffer_size is
    provided, raise a ValueError if the number of bytes consumed by the returned
    array exceeds this value.

    Parameters
    ----------
    buf : ndarray-like, array-like, or bytes-like
        A numpy array like object such as numpy.ndarray, cupy.ndarray, or
        any object exporting a buffer interface.
    max_buffer_size : int
        If specified, the largest allowable value of arr.nbytes, where arr
        is the returned array.
    flatten : bool
        If True, the array are flatten.

    Returns
    -------
    arr : NDArrayLike
        A ndarray-like, sharing memory with `buf`.

    Notes
    -----
    This function will not create a copy under any circumstances, it is guaranteed to
    return a view on memory exported by `buf`.
    """
    arr = ensure_ndarray_like(buf)

    # check for object arrays, these are just memory pointers, actual memory holding
    # item data is scattered elsewhere
    if arr.dtype.kind == "O":
        raise TypeError("object arrays are not supported")

    # check for datetime or timedelta ndarray, the buffer interface doesn't support those
    if arr.dtype.kind in "Mm":
        arr = arr.view(np.int64)

    # check memory is contiguous, if so flatten
    if arr.flags.c_contiguous or arr.flags.f_contiguous:
        if flatten:
            # can flatten without copy
            arr = arr.reshape(-1, order="A")
    else:
        raise ValueError("an array with contiguous memory is required")

    if max_buffer_size is not None and arr.nbytes > max_buffer_size:
        msg = f"Codec does not support buffers of > {max_buffer_size} bytes"
        raise ValueError(msg)

    return arr


def ensure_contiguous_ndarray(
    buf: BufferLike, max_buffer_size: int | None = None, flatten: bool = True
) -> npt.NDArray[Any]:
    """Convenience function to coerce `buf` to a numpy array, if it is not already a
    numpy array. Also ensures that the returned value exports fully contiguous memory,
    and supports the new-style buffer interface. If the optional max_buffer_size is
    provided, raise a ValueError if the number of bytes consumed by the returned
    array exceeds this value.

    Parameters
    ----------
    buf : array-like or bytes-like
        A numpy array or any object exporting a buffer interface.
    max_buffer_size : int
        If specified, the largest allowable value of arr.nbytes, where arr
        is the returned array.
    flatten : bool
        If True, the array are flatten.

    Returns
    -------
    arr : ndarray
        A numpy array, sharing memory with `buf`.

    Notes
    -----
    This function will not create a copy under any circumstances, it is guaranteed to
    return a view on memory exported by `buf`.
    """

    return ensure_ndarray(
        ensure_contiguous_ndarray_like(buf, max_buffer_size=max_buffer_size, flatten=flatten)
    )


def ensure_bytes(buf: BufferLike) -> bytes:
    """Obtain a bytes object from memory exposed by `buf`."""

    if not isinstance(buf, bytes):
        arr = ensure_ndarray_like(buf)

        # check for object arrays, these are just memory pointers,
        # actual memory holding item data is scattered elsewhere
        if arr.dtype.kind == "O":
            raise TypeError("object arrays are not supported")

        # create bytes
        buf = arr.tobytes(order="A")

    return buf


def ensure_text(s: str | BufferLike, encoding: str = "utf-8") -> str:
    """Return `s` if it is a `str`, else decode the bytes it exposes with `encoding`."""
    if isinstance(s, str):
        return s
    return codecs.decode(ensure_contiguous_ndarray(s), encoding)


@overload
def ndarray_copy[SrcT: BufferLike](src: SrcT, dst: None) -> SrcT: ...
@overload
def ndarray_copy[DstT: BufferLike](src: BufferLike, dst: DstT) -> DstT: ...
def ndarray_copy(src: BufferLike, dst: BufferLike | None) -> BufferLike:
    """Copy the contents of the array from `src` to `dst`, and return `dst`.

    If `dst` is `None`, return `src` unchanged. Otherwise `dst` itself is returned, not
    a view of it, so callers that pass a `bytearray` get that `bytearray` back.
    """

    if dst is None:
        # no-op
        return src

    # ensure ndarray like
    src_arr: NDArrayLike = ensure_ndarray_like(src)
    dst_arr: NDArrayLike = ensure_ndarray_like(dst)

    # flatten source array
    src_arr = src_arr.reshape(-1, order="A")

    # ensure same data type
    if dst_arr.dtype.kind != "O":
        src_arr = src_arr.view(dst_arr.dtype)

    # reshape source to match destination
    if src_arr.shape != dst_arr.shape:
        if dst_arr.flags.f_contiguous:
            order = "F"
        else:
            order = "C"
        src_arr = src_arr.reshape(dst_arr.shape, order=order)

    # copy via numpy
    np.copyto(dst_arr, src_arr)  # type: ignore[arg-type]  # cupy arrays support copyto too

    return dst
