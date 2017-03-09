Release notes
=============

.. _release_0.2.0:

0.2.0
-----

New codecs:

* The :class:`numcodecs.quantize.Quantize` codec, which provides support for reducing the precision
  of floating-point data, has been ported over from Zarr
  (`#28 <https://github.com/alimanfoo/numcodecs/issues/28>`_,
  `#31 <https://github.com/alimanfoo/numcodecs/issues/31>`_).

Other changes:

* The :class:`numcodecs.zlib.Zlib` codec is now also registered under the alias 'gzip'
  (`#29 <https://github.com/alimanfoo/numcodecs/issues/29>`_,
  `#32 <https://github.com/alimanfoo/numcodecs/issues/32>`_).

Maintenance work:

* A data fixture has been added to the test suite to add some protection against changes to codecs
  that break backwards-compatibility with data encoded using a previous release of numcodecs
  (`#30 <https://github.com/alimanfoo/numcodecs/issues/30>`_,
  `#33 <https://github.com/alimanfoo/numcodecs/issues/33>`_).

.. _release_0.1.1:

0.1.1
-----

This release includes a small modification to the setup.py script to provide greater control over
how compiler options for different instruction sets are configured
(`#24 <https://github.com/alimanfoo/numcodecs/issues/24>`_,
`#27 <https://github.com/alimanfoo/numcodecs/issues/27>`_).

.. _release_0.1.0:

0.1.0
-----

New codecs:

* Two new compressor codecs :class:`numcodecs.zstd.Zstd` and :class:`numcodecs.lz4.LZ4`
  have been added (`#3 <https://github.com/alimanfoo/numcodecs/issues/3>`_,
  `#22 <https://github.com/alimanfoo/numcodecs/issues/22>`_). These provide direct support for
  compression/decompression using `Zstandard <https://github.com/facebook/zstd>`_ and
  `LZ4 <https://github.com/lz4/lz4>`_ respectively.
* A new :class:`numcodecs.msgpacks.MsgPack` codec has been added which uses
  `msgpack-python <https://github.com/msgpack/msgpack-python>`_ to perform encoding/decoding,
  including support for arrays of Python objects
  (`Jeff Reback <https://github.com/jreback>`_;
  `#5 <https://github.com/alimanfoo/numcodecs/issues/5>`_,
  `#6 <https://github.com/alimanfoo/numcodecs/issues/6>`_,
  `#8 <https://github.com/alimanfoo/numcodecs/issues/8>`_,
  `#21 <https://github.com/alimanfoo/numcodecs/issues/21>`_).
* A new :class:`numcodecs.pickles.Pickle` codec has been added which uses the Python pickle protocol
  to perform encoding/decoding, including support for arrays of Python objects
  (`Jeff Reback <https://github.com/jreback>`_;
  `#5 <https://github.com/alimanfoo/numcodecs/issues/5>`_,
  `#6 <https://github.com/alimanfoo/numcodecs/issues/6>`_,
  `#21 <https://github.com/alimanfoo/numcodecs/issues/21>`_).
* A new :class:`numcodecs.astype.AsType` codec has been added which uses NumPy to perform type
  conversion
  (`John Kirkham <https://github.com/jakirkham>`_;
  `#7 <https://github.com/alimanfoo/numcodecs/issues/7>`_,
  `#12 <https://github.com/alimanfoo/numcodecs/issues/12>`_,
  `#14 <https://github.com/alimanfoo/numcodecs/issues/14>`_).

Other new features:

* The :class:`numcodecs.lzma.LZMA` codec is now supported on Python 2.7 if
  `backports.lzma <https://pypi.python.org/pypi/backports.lzma>`_ is installed
  (`John Kirkham <https://github.com/jakirkham>`_;
  `#11 <https://github.com/alimanfoo/numcodecs/issues/11>`_,
  `#13 <https://github.com/alimanfoo/numcodecs/issues/13>`_).
* The bundled c-blosc library has been upgraded to version
  `1.11.2 <https://github.com/Blosc/c-blosc/releases/tag/v1.11.2>`_
  (`#10 <https://github.com/alimanfoo/numcodecs/issues/10>`_,
  `#18 <https://github.com/alimanfoo/numcodecs/issues/18>`_).
* An option has been added to the :class:`numcodecs.blosc.Blosc` codec to allow the block size to
  be manually configured
  (`#9 <https://github.com/alimanfoo/numcodecs/issues/9>`_,
  `#19 <https://github.com/alimanfoo/numcodecs/issues/19>`_).
* The representation string for the :class:`numcodecs.blosc.Blosc` codec has been tweaked to
  help with understanding the shuffle option
  (`#4 <https://github.com/alimanfoo/numcodecs/issues/4>`_,
  `#19 <https://github.com/alimanfoo/numcodecs/issues/19>`_).
* Options have been added to manually control how the C extensions are built regardless of the
  architecture of the system on which the build is run. To disable support for AVX2 set the
  environment variable "DISABLE_NUMCODECS_AVX2". To disable support for SSE2 set the environment
  variable "DISABLE_NUMCODECS_SSE2". To disable C extensions altogether set the environment variable
  "DISABLE_NUMCODECS_CEXT"
  (`#24 <https://github.com/alimanfoo/numcodecs/issues/24>`_,
  `#26 <https://github.com/alimanfoo/numcodecs/issues/26>`_).

Maintenance work:

* CI tests now run under Python 3.6 as well as 2.7, 3.4, 3.5
  (`#16 <https://github.com/alimanfoo/numcodecs/issues/16>`_,
  `#17 <https://github.com/alimanfoo/numcodecs/issues/17>`_).
* Test coverage is now monitored via
  `coveralls <https://coveralls.io/github/alimanfoo/numcodecs?branch=master>`_
  (`#15 <https://github.com/alimanfoo/numcodecs/issues/15>`_,
  `#20 <https://github.com/alimanfoo/numcodecs/issues/20>`_).

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
