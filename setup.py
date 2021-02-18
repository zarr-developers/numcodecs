from glob import glob
import os
from setuptools import setup
import sys

# setup common compile arguments


def info(*msg):
    kwargs = dict(file=sys.stdout)
    print('[numcodecs]', *msg, **kwargs)


def error(*msg):
    kwargs = dict(file=sys.stderr)
    print('[numcodecs]', *msg, **kwargs)



class BuildFailed(Exception):
    pass


DESCRIPTION = ("A Python package providing buffer compression and "
               "transformation codecs for use in data storage and "
               "communication applications.")

with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()


def run_setup(with_extensions=False):

    setup(
        name='numcodecs',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        use_scm_version={
            'version_scheme': 'guess-next-dev',
            'local_scheme': 'dirty-tag',
            'write_to': 'numcodecs/version.py'
        },
        setup_requires=[
            'setuptools>18.0',
            'setuptools-scm>1.5.4'
        ],
        install_requires=[
            'numpy>=1.7',
        ],
        extras_require={
            'msgpack':  ["msgpack"],
        },
        package_dir={"": "."},
        python_requires=">=3.6, <4",
        packages=["numcodecs", "numcodecs.tests"],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Information Technology",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Operating System :: Unix",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
        author='Alistair Miles',
        author_email='alimanfoo@googlemail.com',
        maintainer='Alistair Miles',
        maintainer_email='alimanfoo@googlemail.com',
        url='https://github.com/zarr-developers/numcodecs',
        license='MIT',
    )


if __name__ == '__main__':
    run_setup()
