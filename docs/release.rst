Release notes
=============

.. _release_0.5.5:

0.5.5
-----

* The bundled c-blosc sources have been upgraded to version 1.14.3 (:issue:`72`).


.. _release_0.5.4:

0.5.4
-----

* The bundled c-blosc sources have been upgraded to version 1.14.0 (:issue:`71`).


.. _release_0.5.3:

0.5.3
-----

* The test suite has been migrated to use pytest instead of nosetests
  (:issue:`61`, :issue:`62`).

* The bundled c-blosc library has been updated to version 1.13.4 (:issue:`63`,
  :issue:`64`).


.. _release_0.5.2:

0.5.2
-----

* Add support for encoding None values in VLen... codecs (:issue:`59`).

 
.. _release_0.5.1:

0.5.1
-----

* Fixed a compatibility issue with the Zlib codec to ensure it can handle
  bytearray objects under Python 2.7 (:issue:`57`).
* Restricted the :class:`numcodecs.categorize.Categorize` codec to object
  ('O') and unicode ('U') dtypes and disallowed bytes ('S') dtypes because
  these do not round-trip through JSON configuration.


.. _release_0.5.0:

0.5.0
-----

* Added new codecs for encoding arrays with variable-length unicode strings
  (:class:`numcodecs.vlen.VLenUTF8`), variable-length byte strings
  (:class:`numcodecs.vlen.VLenBytes`) and variable-length numerical arrays
  ((:class:`numcodecs.vlen.VLenArray`) (:issue:`56`).


.. _release_0.4.1:

0.4.1
-----

* Resolved an issue where providing an array with dtype ``object`` as the destination
  when decoding could cause segaults with some codecs (:issue:`55`).


.. _release_0.4.0:

0.4.0
-----

* Added a new :class:`numcodecs.json.JSON` codec as an alternative for encoding of
  object arrays (:issue:`54`).


.. _release_0.3.1:

0.3.1
-----

* Revert the default shuffle argument to SHUFFLE (byte shuffle) for the
  :class:`numcodecs.blosc.Blosc` codec for compatibility and consistency with previous
  code.


.. _release_0.3.0:

0.3.0
-----

* The :class:`numcodecs.blosc.Blosc` codec has been made robust for usage in both
  multithreading and multiprocessing programs, regardless of whether Blosc has been
  configured to use multiple threads internally or not (:issue:`41`, :issue:`42`).

* The :class:`numcodecs.blosc.Blosc` codec now supports an ``AUTOSHUFFLE`` argument
  when encoding (compressing) which activates bit- or byte-shuffle depending on the
  itemsize of the incoming buffer (:issue:`37`, :issue:`42`). This is also now the
  default.

* The :class:`numcodecs.blosc.Blosc` codec now raises an exception when an invalid
  compressor name is provided under all circumstances (:issue:`40`, :issue:`42`).

* The bundled version of the c-blosc library has been upgraded to version 1.12.1
  (:issue:`45`, :issue:`42`).

* An improvement has been made to the system detection capabilities during compilation
  of C extensions (by :user:`Prakhar Goel <newt0311>`; :issue:`36`, :issue:`38`).

* Arrays with datetime64 or timedelta64 can now be passed directly to compressor codecs
  (:issue:`39`, :issue:`46`).


.. _release_0.2.1:

0.2.1
-----

The bundled c-blosc libary has been upgraded to version 1.11.3 (:issue:`34`, :issue:`35`).


.. _release_0.2.0:

0.2.0
-----

New codecs:

* The :class:`numcodecs.quantize.Quantize` codec, which provides support for reducing the precision
  of floating-point data, has been ported over from Zarr (:issue:`28`, :issue:`31`).

Other changes:

* The :class:`numcodecs.zlib.Zlib` codec is now also registered under the alias 'gzip'
  (:issue:`29`, :issue:`32`).

Maintenance work:

* A data fixture has been added to the test suite to add some protection against changes to codecs
  that break backwards-compatibility with data encoded using a previous release of numcodecs
  (:issue:`30`, :issue:`33`).


.. _release_0.1.1:

0.1.1
-----

This release includes a small modification to the setup.py script to provide greater control over
how compiler options for different instruction sets are configured (:issue:`24`,
:issue:`27`).


.. _release_0.1.0:

0.1.0
-----

New codecs:

* Two new compressor codecs :class:`numcodecs.zstd.Zstd` and :class:`numcodecs.lz4.LZ4`
  have been added (:issue:`3`, :issue:`22`). These provide direct support for
  compression/decompression using `Zstandard <https://github.com/facebook/zstd>`_ and
  `LZ4 <https://github.com/lz4/lz4>`_ respectively.

* A new :class:`numcodecs.msgpacks.MsgPack` codec has been added which uses
  `msgpack-python <https://github.com/msgpack/msgpack-python>`_ to perform encoding/decoding,
  including support for arrays of Python objects
  (`Jeff Reback <https://github.com/jreback>`_; :issue:`5`, :issue:`6`, :issue:`8`,
  :issue:`21`).

* A new :class:`numcodecs.pickles.Pickle` codec has been added which uses the Python pickle protocol
  to perform encoding/decoding, including support for arrays of Python objects
  (`Jeff Reback <https://github.com/jreback>`_; :issue:`5`, :issue:`6`, :issue:`21`).

* A new :class:`numcodecs.astype.AsType` codec has been added which uses NumPy to perform type
  conversion (`John Kirkham <https://github.com/jakirkham>`_; :issue:`7`, :issue:`12`,
  :issue:`14`).

Other new features:

* The :class:`numcodecs.lzma.LZMA` codec is now supported on Python 2.7 if
  `backports.lzma <https://pypi.python.org/pypi/backports.lzma>`_ is installed
  (`John Kirkham <https://github.com/jakirkham>`_; :issue:`11`, :issue:`13`).

* The bundled c-blosc library has been upgraded to version
  `1.11.2 <https://github.com/Blosc/c-blosc/releases/tag/v1.11.2>`_ (:issue:`10`,
  :issue:`18`).

* An option has been added to the :class:`numcodecs.blosc.Blosc` codec to allow the block size to
  be manually configured (:issue:`9`, :issue:`19`).

* The representation string for the :class:`numcodecs.blosc.Blosc` codec has been tweaked to
  help with understanding the shuffle option (:issue:`4`, :issue:`19`).

* Options have been added to manually control how the C extensions are built regardless of the
  architecture of the system on which the build is run. To disable support for AVX2 set the
  environment variable "DISABLE_NUMCODECS_AVX2". To disable support for SSE2 set the environment
  variable "DISABLE_NUMCODECS_SSE2". To disable C extensions altogether set the environment variable
  "DISABLE_NUMCODECS_CEXT" (:issue:`24`, :issue:`26`).

Maintenance work:

* CI tests now run under Python 3.6 as well as 2.7, 3.4, 3.5 (:issue:`16`, :issue:`17`).

* Test coverage is now monitored via
  `coveralls <https://coveralls.io/github/alimanfoo/numcodecs?branch=master>`_
  (:issue:`15`, :issue:`20`).


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
