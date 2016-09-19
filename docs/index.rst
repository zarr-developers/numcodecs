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
library. Installing via conda will install a pre-compiled binary distribution.
However, if you have a newer CPU that supports the AVX2 instruction set (e.g.,
Intel Haswell, Broadwell or Skylake) then installing via pip is preferable,
because this will compile the Blosc library from source with optimisations
for AVX2.

To work with Numcodecs source code in development, install from GitHub::

    $ git clone --recursive https://github.com/alimanfoo/numcodecs.git
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
    blosc
    zlib
    bz2
    lzma
    delta
    fixedscaleoffset
    packbits
    categorize
    checksum32
    registry
    release

Acknowledgments
---------------

Numcodecs bundles the `c-blosc <https://github.com/Blosc/c-blosc>`_
library.

Development of this package is supported by the
`MRC Centre for Genomics and Global Health <http://www.cggh.org>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Blosc: http://www.blosc.org/
