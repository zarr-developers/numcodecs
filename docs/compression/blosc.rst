Blosc
=====
.. automodule:: numcodecs.blosc

.. autoclass:: Blosc

    .. autoattribute:: codec_id
    .. autoattribute:: NOSHUFFLE
    .. autoattribute:: SHUFFLE
    .. autoattribute:: BITSHUFFLE
    .. autoattribute:: AUTOSHUFFLE
    .. automethod:: encode
    .. automethod:: decode
    .. automethod:: get_config
    .. automethod:: from_config
    .. automethod:: decode_partial

Helper functions
----------------

.. autofunction:: list_compressors
.. autofunction:: get_nthreads
.. autofunction:: set_nthreads
.. autofunction:: cbuffer_complib
.. autofunction:: compress
.. autofunction:: decompress
