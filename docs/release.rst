Release notes
=============

.. _unreleased:

Unreleased
----------

.. _release_0.8.1:

0.8.1
-----

* Fix an ImportError with Blosc on Android.
  By :user:`Daniel Jewell <danieldjewell>`, :issue:`284`.

.. _release_0.8.0:

0.8.0
-----

* The :class:`numcodecs.zfpy.ZFPY` codec is now supported on Python 3.8 if
  `zfpy==0.5.5 <https://pypi.org/project/zfpy/>`_ is installed.
  By :user:`haiying xu <halehawk>`, :issue:`229`.

* Add support for byte Shuffle filter
  By :user:`Paul Branson <pbranson>` and :user:`Martin Durant <martindurant>` :issue:`273`.

* Update Windows + Mac CI to run all tests.
  By :user:`Jackson Maxfield Brown <JacksonMaxfield>`, :issue:`276`.
  Help from :user:`Oleg Höfling <hoefling>`, :issue:`273`.

* Update cpuinfo to 8.0.0.
  By :user:`Florian Jetter <fjetter>`, :issue:`280`.

* Drop out-of-date manual release docs.
  By :user:`John Kirkham <jakirkham>`, :issue:`272`.

* Add support for Python 3.9 and Update GitHub Actions.

.. _release_0.7.3:

0.7.3
-----

* Add support for Python 3.9 and Update GitHub Actions.
  By :user:`Jackson Maxfield Brown <JacksonMaxfield>`, :issue:`270`.

* Remove support for Python 3.5 which is end of life. While the code base might
  still be compatible; the source dist and wheel are marked as Python 3.6+ and
  pip will not install them. Continuous integration on Python 3.5 has been
  disabled.
  By :user:`Matthias Bussonnier <Carreau>`, :issue:`266` and :issue:`267`.

.. _release_0.7.2:

0.7.2
-----

* Disable avx2 for wheel.
  By :user:`Grzegorz Bokota <Czaki>`, :issue:`253`.

* Add Base64 fixtures.
  By :user:`John Kirkham <jakirkham>`, :issue:`251`.

* Update docs regarding wheels.
  By :user:`Josh Moore <joshmoore>`, :issue:`250`.


.. _release_0.7.1:

0.7.1
-----

* Fix build of wheels.
  By :user:`Grzegorz Bokota <Czaki>`, :issue:`244`.

.. _release_0.7.0:

0.7.0
-----

* Automatically release to PyPI.
  By :user:`Josh Moore <joshmoore>`, :issue:`241`.

* Build wheels on github actions.
  By :user:`Grzegorz Bokota <Czaki>`, :issue:`224`.

* Add Base64 codec.
  By :user:`Trevor Manz <manzt>`, :issue:`176`.

* Add partial decompression of Blosc compressed arrays.
  By :user:`Andrew Fulton <andrewfulton9>`, :issue:`235`.

* Remove LegacyJSON codec.
  By :user:`James Bourbeau  <jrbourbeau>`, :issue:`226`.

* Remove LegacyMsgPack codec.
  By :user:`James Bourbeau  <jrbourbeau>`, :issue:`218`.

* Drop support for Python 2.
  By :user:`James Bourbeau <jrbourbeau>`, :issue:`220`.


.. _release_0.6.4:

0.6.4
-----

* Update Cython to 0.29.14.
  By :user:`John Kirkham <jakirkham>`, :issue:`168`, :issue:`177`, :issue:`204`.

* The bundled c-blosc sources have been upgraded to version 1.17.0.
  This fixes compilation with newer versions of gcc.
  By :user:`Joe Jevnik <llllllllll>`, :issue:`194`.

* Create ``.pep8speaks.yml``. By :user:`Alistair Miles <alimanfoo>`.

* Simplify datetime/timedelta check.
  By :user:`John Kirkham <jakirkham>`, :issue:`170`, :issue:`171`.

* Update URL metadata for PyPI.
  By :user:`Elliott Sales de Andrade <QuLogic>`, :issue:`178`.

* Enable pytest rewriting in test helper functions.
  By :user:`Elliott Sales de Andrade <QuLogic>`, :issue:`185`.

* Rewrites the ``ensure_text`` implementation.
  By :user:`John Kirkham <jakirkham>`, :issue:`201`, :issue:`205`, :issue:`206`.

* Add macOS to CI.
  By :user:`Alistair Miles <alimanfoo>`, :issue:`192`.

* Fix test failures on big-endian systems.
  By :user:`Elliott Sales de Andrade <QuLogic>`, :issue:`186`.

* Use unittest.mock on Python 3.
  By :user:`Elliott Sales de Andrade <QuLogic>`, :issue:`179`.

* Don't mask compile errors in setup.py.
  By :user:`Joe Jevnik <llllllllll>`, :issue:`197`.

* Allow pickles when loading test fixture data.
  By :user:`Elliott Sales de Andrade <QuLogic>`, :issue:`193`.

* Update ``cpuinfo.py``.
  By :user:`John Kirkham <jakirkham>`, :issue:`202`.

* Use ``ensure_text`` in JSON codecs.
  By :user:`John Kirkham <jakirkham>`, :issue:`207`.

* Support Python 3.8.
  By :user:`John Kirkham <jakirkham>`, :issue:`208`.


.. _release_0.6.3:

0.6.3
-----

* Drop support for 32-bit Windows.
  By :user:`Alistair Miles <alimanfoo>`, :issue:`97`, :issue:`156`.

* Raise a ``TypeError`` if an ``object`` array is passed to ``ensure_bytes``.
  By :user:`John Kirkham <jakirkham>`, :issue:`162`.

* Update Cython to 0.29.3.
  By :user:`John Kirkham <jakirkham>`, :issue:`165`.


.. _release_0.6.2:

0.6.2
-----

* Handle (new) buffer protocol conforming types in ``Pickle.decode``.
  By :user:`John Kirkham <jakirkham>`, :issue:`143`, :issue:`150`.

* Use (new) buffer protocol in ``MsgPack`` codec `decode()` method.
  By :user:`John Kirkham <jakirkham>`, :issue:`148`.

* Use (new) buffer protocol in ``JSON`` codec `decode()` method.
  By :user:`John Kirkham <jakirkham>`, :issue:`151`.

* Avoid copying into data in ``GZip``'s `decode()` method on Python 2.
  By :user:`John Kirkham <jakirkham>`, :issue:`152`.

* Revert ndarray coercion of encode returned data.
  By :user:`John Kirkham <jakirkham>`, :issue:`155`.

* The bundled c-blosc sources have been upgraded to version 1.15.0. By
  :user:`Alistair Miles <alimanfoo>` and :user:`John Kirkham <jakirkham>`, :issue:`142`, :issue:`145`.

.. _release_0.6.1:

0.6.1
-----

* Resolved minor issue in backwards-compatibility tests (by :user:`Alistair Miles
  <alimanfoo>`, :issue:`138`, :issue:`139`).


.. _release_0.6.0:

0.6.0
-----

* The encoding format used by the :class:`JSON` and :class:`MsgPack` codecs has been
  changed to resolve an issue with correctly encoding and decoding some object arrays.
  Now the encoded data includes the original shape of the array, which enables the
  correct shape to be restored on decoding. The previous encoding format is still
  supported, so that any data encoded using a previous version of numcodecs can still be
  read. Thus no changes to user code and applications should be required, other
  than upgrading numcodecs. By :user:`Jerome Kelleher <jeromekelleher>`; :issue:`74`,
  :issue:`75`.

* Updated the msgpack dependency (by :user:`Jerome Kelleher <jeromekelleher>`;
  :issue:`74`, :issue:`75`).

* Added support for ppc64le architecture by updating `cpuinfo.py` from upstream (by
  :user:`Anand S <anandtrex>`; :issue:`82`).

* Allow :class:`numcodecs.blosc.Blosc` compressor to run on systems where locks are not present (by
  :user:`Marcus Kinsella <mckinsel>`, :issue:`83`; and :user:`Tom White <tomwhite>`,
  :issue:`93`).

* Drop Python 3.4 (by :user:`John Kirkham <jakirkham>`; :issue:`89`).

* Add Python 3.7 (by :user:`John Kirkham <jakirkham>`; :issue:`92`).

* Add codec :class:`numcodecs.gzip.GZip` to replace ``gzip`` alias for ``zlib``,
  which was incorrect (by :user:`Jan Funke <funkey>`; :issue:`87`; and :user:`John Kirkham <jakirkham>`, :issue:`134`).

* Corrects handling of ``NaT`` in ``datetime64`` and ``timedelta64`` in various
  compressors (by :user:`John Kirkham <jakirkham>`; :issue:`127`, :issue:`131`).

* Improvements to the compatibility layer used for normalising inputs to encode
  and decode methods in most codecs. This removes unnecessary memory copies for
  some codecs, and also simplifies the implementation of some codecs, improving
  code readability and maintainability. By :user:`John Kirkham <jakirkham>` and
  :user:`Alistair Miles <alimanfoo>`; :issue:`119`, :issue:`121`, :issue:`128`.

* Return values from encode() and decode() methods are now returned as numpy
  arrays for consistency across codecs. By :user:`John Kirkham <jakirkham>`,
  :issue:`136`.

* Improvements to handling of errors in the :class:`numcodecs.blosc.Blosc` and
  :class:`numcodecs.lz4.LZ4` codecs when the maximum allowed size of an input
  buffer is exceeded. By :user:`Jerome Kelleher <jeromekelleher>`, :issue:`80`,
  :issue:`81`.


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
