Release notes
=============

.. _release_0.1.0:

0.1.0
-----

* Two new compressor codecs :class:`numcodecs.zstd.Zstd` and :class:`numcodecs.lz4.LZ4`
  have been added (`#3 <https://github.com/alimanfoo/numcodecs/issues/3>`_,
  `#22 <https://github.com/alimanfoo/numcodecs/issues/22>`_). These provide direct support for
  compression/decompression using `Zstandard <https://github.com/facebook/zstd>`_ and
  `LZ4 <https://github.com/lz4/lz4>`_ respectively.

.. _release_0.0.1:

0.0.1
-----

Fixed project description in setup.py.

.. _release_0.0.0:

0.0.0
-----

First release. This version is a port of the ``codecs`` module from `Zarr
<http://zarr.readthedocs.io>`_ 2.1.0. The following changes have been made from
the original Zarr module:

* Codec classes have been re-organized into separate modules, mostly one per
  codec class, for ease of maintenance.
* Two new codec classes have been added based on 32-bit checksums:
  :class:`numcodecs.checksum32.CRC32` and :class:`numcodecs.checksum32.Adler32`.
* The Blosc extension has been refactored to remove code duplications related
  to handling of buffer compatibility.
