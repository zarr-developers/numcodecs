import numcodecs


def test_version():
    assert isinstance(numcodecs.__version__, str)
    assert numcodecs.__version__


def test_version_module():
    from numcodecs.version import __version__, version

    assert isinstance(__version__, str)
    assert isinstance(version, str)
    assert __version__ == version
    assert __version__ == numcodecs.__version__
