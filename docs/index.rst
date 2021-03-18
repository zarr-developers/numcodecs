.. numcodecs documentation master file, created by
   sphinx-quickstart on Mon May  2 21:40:09 2016.


Numcodecs
=========

.. automodule:: numcodecs

Installation
------------

Numcodecs depends on NumPy. It is generally best to `install NumPy
<http://docs.scipy.org/doc/numpy/user/install.html>`_ first using
whatever method is most appropriate for you operating system and
Python distribution.

Install from PyPI::

    $ pip install numcodecs

Alternatively, install via conda::

    $ conda install -c conda-forge numcodecs

Numcodecs includes a C extension providing integration with the Blosc_
library. Wheels are available for most platforms.

Installing a wheel or via conda will install a pre-compiled binary distribution.
However, if you have a newer CPU that supports the AVX2 instruction set (e.g.,
Intel Haswell, Broadwell or Skylake) then installing via pip is preferable,
because this will compile the Blosc library from source with optimisations
for AVX2.

Note that if you compile the C extensions on a machine with AVX2 support
you probably then cannot use the same binaries on a machine without AVX2.
To disable compilation with AVX2 support regardless of the machine
architecture::

    $ export DISABLE_NUMCODECS_AVX2=
    $ pip install -v --no-cache-dir --no-binary numcodecs numcodecs

To work with Numcodecs source code in development, install from GitHub::

    $ git clone --recursive https://github.com/zarr-developers/numcodecs.git
    $ cd numcodecs
    $ python setup.py install

To verify that Numcodecs has been fully installed (including the Blosc
extension) run the test suite::

    $ pip install nose
    $ python -m nose -v numcodecs

Contents
--------

.. toctree::
    :maxdepth: 2

    abc
    registry
    blosc
    lz4
    zfpy
    zstd
    zlib
    gzip
    bz2
    lzma
    delta
    fixedscaleoffset
    quantize
    packbits
    categorize
    checksum32
    astype
    json
    pickles
    msgpacks
    vlen
    shuffle
    release
    contributing

Acknowledgments
---------------

The following people have contributed to the development of NumCodecs by contributing code,
documentation, code reviews, comments and/or ideas:

* :user:`Francesc Alted <FrancescAlted>`
* :user:`Prakhar Goel <newt0311>`
* :user:`Jerome Kelleher <jeromekelleher>`
* :user:`John Kirkham <jakirkham>`
* :user:`Alistair Miles <alimanfoo>`
* :user:`Jeff Reback <jreback>`
* :user:`Trevor Manz <manzt>`
* :user:`Grzegorz Bokota <Czaki>`
* :user:`Josh Moore <joshmoore>`
* :user:`Martin Durant <martindurant>`
* :user:`Paul Branson <pbranson>`

Numcodecs bundles the `c-blosc <https://github.com/Blosc/c-blosc>`_ library.

Development of this package is supported by the
`MRC Centre for Genomics and Global Health <http://www.cggh.org>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Blosc: http://www.blosc.org/
