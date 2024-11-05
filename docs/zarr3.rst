Zarr 3 codecs
=============
.. automodule:: numcodecs.zarr3


Bytes-to-bytes codecs
---------------------
.. autoclass:: BloscCodec()

    .. autoattribute:: codec_name

.. autoclass:: Lz4Codec()

    .. autoattribute:: codec_name

.. autoclass:: ZstdCodec()

    .. autoattribute:: codec_name

.. autoclass:: ZlibCodec()

    .. autoattribute:: codec_name

.. autoclass:: GzipCodec()

    .. autoattribute:: codec_name

.. autoclass:: Bz2Codec()

    .. autoattribute:: codec_name

.. autoclass:: LzmaCodec()

    .. autoattribute:: codec_name

.. autoclass:: ShuffleCodec()

    .. autoattribute:: codec_name


Array-to-array codecs
---------------------
.. autoclass:: DeltaCodec()

    .. autoattribute:: codec_name

.. autoclass:: BitroundCodec()

    .. autoattribute:: codec_name

.. autoclass:: FixedScaleOffsetCodec()

    .. autoattribute:: codec_name

.. autoclass:: QuantizeCodec()

    .. autoattribute:: codec_name

.. autoclass:: PackbitsCodec()

    .. autoattribute:: codec_name

.. autoclass:: AsTypeCodec()

    .. autoattribute:: codec_name


Bytes-to-bytes checksum codecs
------------------------------
.. autoclass:: Crc32Codec()

    .. autoattribute:: codec_name

.. autoclass:: Crc32cCodec()

    .. autoattribute:: codec_name

.. autoclass:: Adler32Codec()

    .. autoattribute:: codec_name

.. autoclass:: Fletcher32Codec()

    .. autoattribute:: codec_name

.. autoclass:: JenkinsLookup3Codec()

    .. autoattribute:: codec_name


Array-to-bytes codecs
---------------------
.. autoclass:: PCodecCodec()

    .. autoattribute:: codec_name

.. autoclass:: ZFPYCodec()

    .. autoattribute:: codec_name
