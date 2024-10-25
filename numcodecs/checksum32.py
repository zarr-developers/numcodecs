import struct
import zlib
from typing import Literal

import numpy as np
from crc32c import crc32c as crc32c_

from .abc import Codec
from .compat import ensure_contiguous_ndarray, ndarray_copy
from .jenkins import jenkins_lookup3

CHECKSUM_LOCATION = Literal['start', 'end']


class Checksum32(Codec):
    # override in sub-class
    checksum = None
    location: CHECKSUM_LOCATION = 'start'

    def __init__(self, location: CHECKSUM_LOCATION | None = None):
        if location is not None:
            self.location = location

    def encode(self, buf):
        arr = ensure_contiguous_ndarray(buf).view('u1')
        checksum = self.checksum(arr) & 0xFFFFFFFF
        enc = np.empty(arr.nbytes + 4, dtype='u1')
        if self.location == 'start':
            checksum_view = enc[:4]
            payload_view = enc[4:]
        elif self.location == 'end':
            checksum_view = enc[-4:]
            payload_view = enc[:-4]
        else:
            raise ValueError(f"Invalid checksum location: {self.location}")
        checksum_view.view('<u4')[0] = checksum
        ndarray_copy(arr, payload_view)
        return enc

    def decode(self, buf, out=None):
        if len(buf) < 4:
            raise ValueError("Input buffer is too short to contain a 32-bit checksum.")
        if out is not None:
            ensure_contiguous_ndarray(out)  # check that out is a valid ndarray

        arr = ensure_contiguous_ndarray(buf).view('u1')
        if self.location == 'start':
            checksum_view = arr[:4]
            payload_view = arr[4:]
        elif self.location == 'end':
            checksum_view = arr[-4:]
            payload_view = arr[:-4]
        else:
            raise ValueError(f"Invalid checksum location: {self.location}")
        expect = checksum_view.view('<u4')[0]
        checksum = self.checksum(payload_view) & 0xFFFFFFFF
        if expect != checksum:
            raise RuntimeError(
                f"Stored and computed {self.codec_id} checksum do not match. Stored: {expect}. Computed: {checksum}."
            )
        return ndarray_copy(payload_view, out)


class CRC32(Checksum32):
    codec_id = 'crc32'
    checksum = zlib.crc32


class CRC32C(Checksum32):
    codec_id = 'crc32c'
    checksum = crc32c_
    location = 'end'


class Adler32(Checksum32):
    codec_id = 'adler32'
    checksum = zlib.adler32


class JenkinsLookup3(Checksum32):
    """Bob Jenkin's lookup3 checksum with 32-bit output

    This is the HDF5 implementation.
    https://github.com/HDFGroup/hdf5/blob/577c192518598c7e2945683655feffcdbdf5a91b/src/H5checksum.c#L378-L472

    With this codec, the checksum is concatenated on the end of the data
    bytes when encoded. At decode time, the checksum is performed on
    the data portion and compared with the four-byte checksum, raising
    RuntimeError if inconsistent.

    Attributes:
       initval: initial seed passed to the hash algorithm, default: 0
       prefix: bytes prepended to the buffer before evaluating the hash, default: None
    """

    checksum = jenkins_lookup3
    codec_id = "jenkins_lookup3"

    def __init__(self, initval: int = 0, prefix=None):
        self.initval = initval
        if prefix is None:
            self.prefix = None
        else:
            self.prefix = np.frombuffer(prefix, dtype='uint8')

    def encode(self, buf):
        """Return buffer plus 4-byte Bob Jenkin's lookup3 checksum"""
        buf = ensure_contiguous_ndarray(buf).ravel().view('uint8')
        if self.prefix is None:
            val = jenkins_lookup3(buf, self.initval)
        else:
            val = jenkins_lookup3(np.hstack((self.prefix, buf)), self.initval)
        return buf.tobytes() + struct.pack("<I", val)

    def decode(self, buf, out=None):
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
            out.view("uint8")[:] = b[:-4]
            return out
        return memoryview(b[:-4])
