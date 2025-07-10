Zstd
====
.. automodule:: numcodecs.zstd

.. autoclass:: Zstd

    .. autoattribute:: codec_id
    .. automethod:: encode
    .. automethod:: decode
        .. note::
            If the compressed data does not contain the decompressed size, streaming
            decompression will be used.
    .. automethod:: get_config
    .. automethod:: from_config

Helper functions
----------------

.. autofunction:: compress
.. autofunction:: decompress
