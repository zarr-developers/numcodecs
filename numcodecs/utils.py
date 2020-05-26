from contextlib import contextmanager
import warnings

class NumcodecsUnstableWarning(FutureWarning):
    """
    Exception raise by an experimental feature in this numcodec

    Wrap code in :any:`allow_unstable` to let code dirrectly (or indirectly),
    use unstable features. 
    """

warnings.filterwarnings('error', category=NumcodecsUnstableWarning)

@contextmanager
def allow_unstable(action='default'):
    """
    This context manager has to be used in any place where unstable completer
    behavior and API may be called.

    >>> with allow_unstable():
    ...     numcodecs.do_experimental_things() # works

    >>> numcodecs.do_experimental_things() # raises.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(action, category=NumcodecsUnstableWarning)
        yield


def unstable():
    warnings.warn(
            "You are (in)directly using an unstable feature, use the "
            "``allow_unstable()`` context manager."
    , category=NumcodecsUnstableWarning, stacklevel=2)

