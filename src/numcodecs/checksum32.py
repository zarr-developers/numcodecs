import abc
import importlib.util
import struct
import warnings
import zlib
from collections.abc import Buffer, Callable
from typing import Literal, NotRequired, cast, overload

import numpy as np
import numpy.typing as npt

from .abc import Codec, CodecConfig
from .compat import ensure_contiguous_ndarray, ndarray_copy
from .jenkins import jenkins_lookup3
from .ndarray_like import BufferLike

crc32c_checksum: Callable[[Buffer, int], int] | None

warnings.filterwarnings('once', message='crc32c usage is deprecated.*', category=DeprecationWarning)

if importlib.util.find_spec("google_crc32c") is not None:
    import google_crc32c

    def crc32c_checksum(data: Buffer, value: int = 0) -> int:
        result: int = google_crc32c.value(data) if value == 0 else google_crc32c.extend(value, data)
        return result

elif importlib.util.find_spec("crc32c") is not None:
    warnings.warn(
        "crc32c usage is deprecated since numcodecs v0.16.4. "
        "It is recommended to install google_crc32c instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    import crc32c

    def crc32c_checksum(data: Buffer, value: int = 0) -> int:
        return crc32c.crc32c(data, value)
else:
    crc32c_checksum = None

CHECKSUM_LOCATION = Literal['start', 'end']


class Checksum32Config(CodecConfig):
    """The configuration of a `Checksum32` codec (CRC32, Adler32, CRC32C), as returned by
    `get_config`. `location` is present only when it was passed to the constructor; the
    class default applies otherwise (`'start'` for CRC32 and Adler32, `'end'` for
    CRC32C)."""

    location: NotRequired[CHECKSUM_LOCATION]


class JenkinsLookup3Config(Checksum32Config):
    """The configuration of a `JenkinsLookup3` codec, as returned by
    `JenkinsLookup3.get_config`."""

    initval: int
    prefix: list[int] | None


class Checksum32(Codec, abc.ABC):
    # override in sub-class
    location: CHECKSUM_LOCATION = 'start'

    def __init__(self, location: CHECKSUM_LOCATION | None = None) -> None:
        if location is not None:
            self.location = location
        if self.location not in ['start', 'end']:
            raise ValueError(f"Invalid checksum location: {self.location}")

    def get_config(self) -> Checksum32Config:
        return cast(Checksum32Config, super().get_config())

    # CRC32, Adler32 and CRC32C return uint8 ndarrays; JenkinsLookup3 overrides encode and
    # decode to return bytes and a memoryview. The annotations here cover both.
    def encode(self, buf: BufferLike) -> npt.NDArray[np.uint8] | bytes:
        arr = ensure_contiguous_ndarray(buf).view('u1')
        checksum = self.checksum(arr) & 0xFFFFFFFF
        enc = np.empty(arr.nbytes + 4, dtype='u1')
        if self.location == 'start':
            checksum_view = enc[:4]
            payload_view = enc[4:]
        else:
            checksum_view = enc[-4:]
            payload_view = enc[:-4]
        checksum_view.view('<u4')[0] = checksum
        ndarray_copy(arr, payload_view)
        return enc

    @overload
    def decode(self, buf: BufferLike, out: None = None) -> npt.NDArray[np.uint8] | memoryview: ...
    @overload
    def decode[OutT: BufferLike](self, buf: BufferLike, out: OutT) -> OutT: ...
    def decode(self, buf: BufferLike, out: BufferLike | None = None) -> BufferLike:
        arr = ensure_contiguous_ndarray(buf).view('u1')
        if arr.nbytes < 4:
            raise ValueError("Input buffer is too short to contain a 32-bit checksum.")
        if out is not None:
            ensure_contiguous_ndarray(out)  # check that out is a valid ndarray

        if self.location == 'start':
            checksum_view = arr[:4]
            payload_view = arr[4:]
        else:
            checksum_view = arr[-4:]
            payload_view = arr[:-4]
        expect = checksum_view.view('<u4')[0]
        checksum = self.checksum(payload_view) & 0xFFFFFFFF
        if expect != checksum:
            raise RuntimeError(
                f"Stored and computed {self.codec_id} checksum do not match. Stored: {expect}. Computed: {checksum}."
            )
        return ndarray_copy(payload_view, out)

    @staticmethod
    @abc.abstractmethod
    def checksum(data: Buffer, value: int = 0, /) -> int:
        """Return the checksum of `data`, continuing from `value`. Subclasses choose the
        default `value`."""


class CRC32(Checksum32):
    """Codec add a crc32 checksum to the buffer.

    Parameters
    ----------
    location : 'start' or 'end'
        Where to place the checksum in the buffer.
    """

    codec_id = 'crc32'
    location: CHECKSUM_LOCATION = 'start'

    @staticmethod
    def checksum(data: Buffer, value: int = 0) -> int:
        """
        Thin wrapper around ``zlib.crc32``.
        """
        return zlib.crc32(data, value)


class Adler32(Checksum32):
    """Codec add a adler32 checksum to the buffer.

    Parameters
    ----------
    location : 'start' or 'end'
        Where to place the checksum in the buffer.
    """

    codec_id = 'adler32'
    location: CHECKSUM_LOCATION = 'start'

    @staticmethod
    def checksum(data: Buffer, value: int = 1) -> int:
        """
        Thin wrapper around ``zlib.adler32``.
        """
        return zlib.adler32(data, value)


class JenkinsLookup3(Checksum32):
    """Bob Jenkin's lookup3 checksum with 32-bit output

    This is the HDF5 implementation.
    https://github.com/HDFGroup/hdf5/blob/577c192518598c7e2945683655feffcdbdf5a91b/src/H5checksum.c#L378-L472

    With this codec, the checksum is concatenated on the end of the data
    bytes when encoded. At decode time, the checksum is performed on
    the data portion and compared with the four-byte checksum, raising
    RuntimeError if inconsistent.

    Parameters
    ----------
    initval : int
        initial seed passed to the hash algorithm, default: 0
    prefix : bytes-like or list of int
        bytes prepended to the buffer before evaluating the hash, default: None.
        `get_config` reports them as a list of ints, so that the configuration is JSON.
    """

    checksum = staticmethod(jenkins_lookup3)
    codec_id = "jenkins_lookup3"

    prefix: npt.NDArray[np.uint8] | None

    def __init__(self, initval: int = 0, prefix: Buffer | list[int] | None = None) -> None:
        self.initval = initval
        if prefix is None:
            self.prefix = None
        else:
            self.prefix = np.frombuffer(bytes(prefix), dtype='uint8')

    def get_config(self) -> "JenkinsLookup3Config":
        config = cast(JenkinsLookup3Config, dict(super().get_config()))
        if self.prefix is not None:
            config["prefix"] = self.prefix.tolist()
        return config

    def encode(self, buf: BufferLike) -> bytes:
        """Return buffer plus 4-byte Bob Jenkin's lookup3 checksum"""
        arr = ensure_contiguous_ndarray(buf).ravel().view('uint8')
        if self.prefix is None:
            val = jenkins_lookup3(arr, self.initval)
        else:
            val = jenkins_lookup3(np.hstack((self.prefix, arr)), self.initval)
        return arr.tobytes() + struct.pack("<I", val)

    @overload
    def decode(self, buf: BufferLike, out: None = None) -> memoryview: ...
    @overload
    def decode[OutT: BufferLike](self, buf: BufferLike, out: OutT) -> OutT: ...
    def decode(self, buf: BufferLike, out: BufferLike | None = None) -> BufferLike:
        """Check Bob Jenkin's lookup3 checksum, and return buffer without it"""
        b = ensure_contiguous_ndarray(buf).view('uint8')
        if self.prefix is None:
            val = jenkins_lookup3(b[:-4], self.initval)
        else:
            val = jenkins_lookup3(np.hstack((self.prefix, b[:-4])), self.initval)
        found = b[-4:].view("<u4")[0]
        if val != found:
            raise RuntimeError(
                f"The Bob Jenkin's lookup3 checksum of the data ({val}) did not"
                f" match the expected checksum ({found}).\n"
                "This could be a sign that the data has been corrupted."
            )
        if out is not None:
            out_view = ensure_contiguous_ndarray(out).view("uint8")
            if out_view.nbytes != b.nbytes - 4:
                raise ValueError(
                    f"out has {out_view.nbytes} bytes, but the decoded data has {b.nbytes - 4}"
                )
            out_view[:] = b[:-4]
            return out
        return memoryview(b[:-4])


if crc32c_checksum is not None:

    class CRC32C(Checksum32):
        """Codec add a crc32c checksum to the buffer.

        Parameters
        ----------
        location : 'start' or 'end'
            Where to place the checksum in the buffer.
        """

        codec_id = 'crc32c'
        location: CHECKSUM_LOCATION = 'end'

        @staticmethod
        def checksum(data: Buffer, value: int = 0) -> int:
            return crc32c_checksum(data, value)  # type: ignore[misc]
