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

.. autofunction:: init
.. autofunction:: destroy
.. autofunction:: compname_to_compcode
.. autofunction:: list_compressors
.. autofunction:: get_nthreads
.. autofunction:: set_nthreads
.. autofunction:: cbuffer_sizes
.. autofunction:: cbuffer_complib
.. autofunction:: cbuffer_metainfo
.. autofunction:: compress
.. autofunction:: decompress
.. autofunction:: decompress_partial
