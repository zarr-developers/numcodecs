# Codec API

```{eval-rst}
.. automodule:: numcodecs.abc

.. autoclass:: Codec

    .. autoattribute:: codec_id
    .. automethod:: encode
    .. automethod:: decode
    .. automethod:: get_config
    .. automethod:: from_config

Protocols
---------

These describe codecs structurally, for code that accepts any object implementing the
interface whether or not it subclasses :class:`Codec`. The three kinds mirror the codec
kinds of the Zarr version 3 specification.

.. autoclass:: CodecProtocol
.. autoclass:: BytesBytesCodec
.. autoclass:: ArrayArrayCodec
.. autoclass:: ArrayBytesCodec
```
