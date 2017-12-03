# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import zlib


import numpy as np


from .abc import Codec
from .compat import ndarray_from_buffer, buffer_copy


class Checksum32(Codec):

    checksum = None

    def encode(self, buf):
        if isinstance(buf, np.ndarray) and buf.dtype == object:
            raise ValueError('cannot encode object array')
        arr = ndarray_from_buffer(buf, dtype='u1')
        checksum = self.checksum(arr) & 0xffffffff
        enc = np.empty(arr.nbytes + 4, dtype='u1')
        enc[:4].view('<u4')[0] = checksum
        enc[4:] = arr
        return enc

    def decode(self, buf, out=None):
        arr = ndarray_from_buffer(buf, dtype='u1')
        expect = arr[:4].view('<u4')[0]
        checksum = self.checksum(arr[4:]) & 0xffffffff
        if expect != checksum:
            raise RuntimeError('checksum failed')
        return buffer_copy(arr[4:], out)


class CRC32(Checksum32):

    codec_id = 'crc32'
    checksum = zlib.crc32


class Adler32(Checksum32):

    codec_id = 'adler32'
    checksum = zlib.adler32
