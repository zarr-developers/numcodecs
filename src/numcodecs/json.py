import json as _json
import textwrap

import numpy as np

from .abc import Codec
from .compat import ensure_text


class JSON(Codec):
    """Codec to encode data as JSON. Useful for encoding an array of Python objects.

    .. versionchanged:: 0.6
        The encoding format has been changed to include the array shape in the encoded
        data, which ensures that all object arrays can be correctly encoded and decoded.

    Examples
    --------
    >>> import numcodecs
    >>> import numpy as np
    >>> x = np.array(['foo', 'bar', 'baz'], dtype='object')
    >>> codec = numcodecs.JSON()
    >>> codec.decode(codec.encode(x))
    array(['foo', 'bar', 'baz'], dtype=object)

    See Also
    --------
    numcodecs.pickles.Pickle, numcodecs.msgpacks.MsgPack

    """

    codec_id = 'json2'

    def __init__(
        self,
        encoding='utf-8',
        skipkeys=False,
        ensure_ascii=True,
        check_circular=True,
        allow_nan=True,
        sort_keys=True,
        indent=None,
        separators=None,
        strict=True,
    ):
        self._text_encoding = encoding
        if separators is None:
            # ensure separators are explicitly specified, and consistent behaviour across
            # Python versions, and most compact representation if indent is None
            if indent is None:
                separators = ',', ':'
            else:
                separators = ', ', ': '
        separators = tuple(separators)
        self._encoder_config = {
            'skipkeys': skipkeys,
            'ensure_ascii': ensure_ascii,
            'check_circular': check_circular,
            'allow_nan': allow_nan,
            'indent': indent,
            'separators': separators,
            'sort_keys': sort_keys,
        }
        self._encoder = _json.JSONEncoder(**self._encoder_config)
        self._decoder_config = {'strict': strict}
        self._decoder = _json.JSONDecoder(**self._decoder_config)

    def encode(self, buf):
        try:
            buf = np.asarray(buf)
        except ValueError:  # pragma: no cover
            buf = np.asarray(buf, dtype=object)
        items = np.atleast_1d(buf).tolist()
        if buf.dtype.fields is not None:
            # tolist() leaves subarray fields as arrays
            items = _arrays_to_lists(items)
            # dtype.str is only "|V<n>" for structured dtypes, so store the fields
            items.append(buf.dtype.descr)
        else:
            items.append(buf.dtype.str)
        items.append(buf.shape)
        return self._encoder.encode(items).encode(self._text_encoding)

    def decode(self, buf, out=None):
        items = self._decoder.decode(ensure_text(buf, self._text_encoding))
        dtype = items[-2]
        if isinstance(dtype, list):
            # structured dtype, encoded as dtype.descr (JSON turned tuples into lists)
            dtype = np.dtype(_descr_from_json(dtype))
            values = _records_from_json(items[:-2], dtype, max(len(items[-1]), 1))
        else:
            values = items[:-2]
        dec = np.empty(items[-1], dtype=dtype)
        if not items[-1]:
            dec[...] = values[0]
        else:
            dec[:] = values
        if out is not None:
            np.copyto(out, dec)
            return out
        else:
            return dec

    def get_config(self):
        config = {'id': self.codec_id, 'encoding': self._text_encoding}
        config.update(self._encoder_config)
        config.update(self._decoder_config)
        return config

    def __repr__(self):
        params = [f'encoding={self._text_encoding!r}']
        for k, v in sorted(self._encoder_config.items()):
            params.append(f'{k}={v!r}')
        for k, v in sorted(self._decoder_config.items()):
            params.append(f'{k}={v!r}')
        classname = type(self).__name__
        params = ', '.join(params)
        return textwrap.fill(
            f'{classname}({params})', width=80, break_long_words=False, subsequent_indent='     '
        )


def _arrays_to_lists(values):
    """Recursively replace the arrays left by ``tolist()`` on a structured array."""
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        return [_arrays_to_lists(v) for v in values]
    return values


def _descr_from_json(descr):
    """Rebuild the list of tuples of a structured ``dtype.descr`` decoded from JSON."""
    fields = []
    for name, field_type, *shape in descr:
        if isinstance(field_type, list):
            field_type = _descr_from_json(field_type)
        fields.append((name, field_type, *(tuple(s) for s in shape)))
    return fields


def _records_from_json(values, dtype, depth=0):
    """Turn the records of a structured array decoded from JSON back into tuples.

    ``depth`` is the number of array dimensions above the records.
    """
    if depth > 0:
        return [_records_from_json(v, dtype, depth - 1) for v in values]
    if dtype.fields is not None:
        return tuple(
            _records_from_json(v, dtype.fields[name][0])
            for name, v in zip(dtype.names, values, strict=True)
        )
    if dtype.subdtype is not None and dtype.subdtype[0].fields is not None:
        base, shape = dtype.subdtype
        return _records_from_json(values, base, len(shape))
    return values
