import numpy.typing as npt


def is_ndarray_like(obj: object) -> bool:
    """Return True when `obj` is ndarray-like"""
    return isinstance(obj, npt.ArrayLike)
