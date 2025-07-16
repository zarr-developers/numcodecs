.. _Zarr 3 codecs:

Zarr 3 codecs
=============
.. automodule:: numcodecs.zarr3


Compressors (bytes-to-bytes codecs)
-----------------------------------
.. autoclass:: Blosc()

    .. autoattribute:: codec_name

.. autoclass:: LZ4()

    .. autoattribute:: codec_name

.. autoclass:: Zstd()

    .. autoattribute:: codec_name

.. autoclass:: Zlib()

    .. autoattribute:: codec_name

.. autoclass:: GZip()

    .. autoattribute:: codec_name

.. autoclass:: BZ2()

    .. autoattribute:: codec_name

.. autoclass:: LZMA()

    .. autoattribute:: codec_name

.. autoclass:: Shuffle()

    .. autoattribute:: codec_name


Checksum codecs (bytes-to-bytes codecs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Need to be used as ``compressors`` in zarr-python.

.. autoclass:: CRC32()

    .. autoattribute:: codec_name

.. autoclass:: CRC32C()

    .. autoattribute:: codec_name

.. autoclass:: Adler32()

    .. autoattribute:: codec_name

.. autoclass:: Fletcher32()

    .. autoattribute:: codec_name

.. autoclass:: JenkinsLookup3()

    .. autoattribute:: codec_name


Filters (array-to-array codecs)
-------------------------------
.. autoclass:: Delta()

    .. autoattribute:: codec_name

.. autoclass:: BitRound()

    .. autoattribute:: codec_name

.. autoclass:: FixedScaleOffset()

    .. autoattribute:: codec_name

.. autoclass:: Quantize()

    .. autoattribute:: codec_name

.. autoclass:: PackBits()

    .. autoattribute:: codec_name

.. autoclass:: AsType()

    .. autoattribute:: codec_name



Serializers (array-to-bytes codecs)
-----------------------------------
.. autoclass:: PCodec()

    .. autoattribute:: codec_name

.. autoclass:: ZFPY()

    .. autoattribute:: codec_name
