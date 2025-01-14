"""
This module defines custom exceptions that are raised in the `numcodecs` codebase.
"""


class UnknownCodecError(Exception):
    """
    An exception that is raised when trying to receive a codec that has not been registered.

    Parameters
    ----------
    codec_id : str
        Codec identifier.
    """

    def __init__(self, codec_id: str):
        self.codec_id = codec_id
        super().__init__(f"codec not available: '{codec_id}'")
