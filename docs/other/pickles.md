# Pickle

```{admonition} Deprecated
:class: warning

The `Pickle` codec is deprecated and will be removed in numcodecs 0.19.0. Constructing it,
directly or through `numcodecs.get_codec({"id": "pickle"})`, emits a `DeprecationWarning`.
Pickled data can only be read by Python, and decoding it can execute arbitrary code. To encode
arrays of Python objects, use one of these codecs instead:

- [`VLenUTF8`](vlen.md) for strings
- [`VLenBytes`](vlen.md) for bytes
- [`VLenArray`](vlen.md) for variable-length numeric arrays
- [`JSON`](json.md) or [`MsgPack`](msgpacks.md) for mixed objects

Data written with `Pickle` can still be read until the codec is removed, subject to the
decoding opt-in described below.
```

```{warning}
Decoding pickled data can execute arbitrary code. `Pickle.decode` raises
`numcodecs.errors.UnsafeDecodeError` unless the process opts in, either by setting
`numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = True` or by setting the environment variable
`NUMCODECS_ALLOW_UNSAFE_PICKLE_DECODE=1` (or `"yes"` / `"true"`) before importing numcodecs. Only
opt in if you trust the source of the data.
```

```{eval-rst}
.. automodule:: numcodecs.pickles

.. autodata:: ALLOW_UNSAFE_PICKLE_DECODE

.. autoclass:: Pickle

    .. autoattribute:: codec_id
    .. automethod:: encode
    .. automethod:: decode
    .. automethod:: get_config
    .. automethod:: from_config
```
