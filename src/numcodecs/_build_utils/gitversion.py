#!/usr/bin/env python3
"""Get the version string for numcodecs, used by meson.build at configure time."""

import os


def get_version():
    try:
        from setuptools_scm import get_version

        return get_version(root=os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    except (ImportError, LookupError):
        pass

    # Fallback: read from pyproject.toml
    pyproject = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pyproject.toml')
    with open(pyproject) as f:
        for line in f:
            if line.strip().startswith('version'):
                return line.split('=')[1].strip().strip('"').strip("'")

    return '0.0.0'


if __name__ == '__main__':
    print(get_version())
