Pickle
======

.. warning::
    Decoding pickled data can execute arbitrary code. ``Pickle.decode`` raises
    ``numcodecs.errors.UnsafeDecodeError`` unless the process opts in, either by setting
    ``numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = True`` or by setting the environment
    variable ``NUMCODECS_ALLOW_UNSAFE_PICKLE_DECODE=1`` (or ``"yes"`` / ``"true"``) before
    importing numcodecs. Only opt in if you trust the source of the data.

.. automodule:: numcodecs.pickles

.. autodata:: ALLOW_UNSAFE_PICKLE_DECODE

.. autoclass:: Pickle

    .. autoattribute:: codec_id
    .. automethod:: encode
    .. automethod:: decode
    .. automethod:: get_config
    .. automethod:: from_config
