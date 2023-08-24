Checksum codecs
===============
.. automodule:: numcodecs.checksum32

CRC32
-----
.. autoclass:: CRC32

    .. autoattribute:: codec_id
    .. automethod:: encode
    .. automethod:: decode
    .. automethod:: get_config
    .. automethod:: from_config


Adler32
-------
.. autoclass:: Adler32

    .. autoattribute:: codec_id
    .. automethod:: encode
    .. automethod:: decode
    .. automethod:: get_config
    .. automethod:: from_config


Fletcher32
----------

.. autoclass:: numcodecs.fletcher32.Fletcher32

    .. autoattribute:: codec_id
    .. automethod:: encode
    .. automethod:: decode

JenkinsLookup3
--------------

.. autoclass:: JenkinsLookup3

    .. autoattribute:: codec_id
    .. autoattribute:: initval
    .. autoattribute:: prefix
    .. automethod:: encode
    .. automethod:: decode
