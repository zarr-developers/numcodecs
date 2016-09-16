# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


class Codec(object):  # pragma: no cover
    """Codec abstract base class."""

    # override in sub-class
    codec_id = None

    def encode(self, buf):
        """Encode data in `buf`.

        Parameters
        ----------
        buf : buffer-like
            Data to be encoded. May be any object supporting the new-style
            buffer protocol or `array.array` (only supports old-style buffer
            protocol in PY2).

        Returns
        -------
        enc : buffer-like
            Encoded data. May be any object supporting the new-style buffer
            protocol or `array.array` (only supports old-style buffer
            protocol in PY2).

        """
        # override in sub-class
        raise NotImplementedError

    def decode(self, buf, out=None):
        """Decode data in `buf`.

        Parameters
        ----------
        buf : buffer-like
            Encoded data. May be any object supporting the new-style buffer
            protocol or `array.array` (only supports old-style buffer
            protocol in PY2).
        out : buffer-like, optional
            Writeable buffer to store decoded data.

        Returns
        -------
        dec : buffer-like
            Decoded data. May be any object supporting the new-style
            buffer protocol or `array.array` (only supports old-style buffer
            protocol in PY2).

        """
        # override in sub-class
        raise NotImplementedError

    def get_config(self):
        """Return a dictionary holding configuration parameters for this
        codec. Must include an 'id' field with the codec ID. All values must be
        compatible with JSON encoding."""
        # override in sub-class if need special encoding of config values
        config = dict(id=self.codec_id)
        # add in all non-private members
        for k in self.__dict__:
            if not k.startswith('_'):
                config[k] = getattr(self, k)
        return config

    @classmethod
    def from_config(cls, config):
        """Instantiate codec from a configuration object."""
        # N.B., assume at this point the 'id' field has been removed from
        # the config object
        # override in sub-class if need special decoding of config values
        return cls(**config)

    def __eq__(self, other):
        # override in sub-class if need special equality comparison
        try:
            return self.get_config() == other.get_config()
        except AttributeError:
            return False

    def __repr__(self):
        # override in sub-class if need special representation
        r = '%s(' % type(self).__name__
        # by default, include all non-private members
        params = ['%s=%r' % (k, getattr(self, k))
                  for k in sorted(self.__dict__)
                  if not k.startswith('_')]
        r += ', '.join(params) + ')'
        return r
