# Codec API

```{eval-rst}
.. automodule:: numcodecs.abc

.. autoclass:: Codec

    .. autoattribute:: codec_id
    .. automethod:: encode
    .. automethod:: decode
    .. automethod:: get_config
    .. automethod:: from_config

Protocol
--------

:class:`CodecProtocol` describes a codec structurally, for code that accepts any object
implementing the interface whether or not it subclasses :class:`Codec`.

.. autoclass:: CodecProtocol
```
