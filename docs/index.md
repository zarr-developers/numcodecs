# Numcodecs

```{eval-rst}
.. automodule:: numcodecs
```

## Installation

Numcodecs depends on NumPy. It is generally best to [install NumPy](https://numpy.org/install/)
first using whatever method is most appropriate for you operating system and Python distribution.

Install from PyPI:

```
$ pip install numcodecs
```

Alternatively, install via conda:

```
$ conda install -c conda-forge numcodecs
```

Numcodecs includes a C extension providing integration with the [Blosc](https://www.blosc.org/)
library. Wheels are available for most platforms.

Installing a wheel or via conda will install a pre-compiled binary distribution.
However, if you have a newer CPU that supports the AVX2 instruction set (e.g.,
Intel Haswell, Broadwell or Skylake) then installing via pip is preferable,
because you can compile the Blosc library from source with optimisations
for AVX2.

```
$ pip install -v --no-cache-dir --no-binary numcodecs numcodecs
```

Note that if you compile the C extensions on a machine with AVX2 support
you probably then cannot use the same binaries on a machine without AVX2.

If you specifically want to disable AVX2 or SSE2 when compiling from source,
you can pass meson build options::

    $ pip install -v --no-cache-dir --no-binary numcodecs numcodecs \
        --config-settings=setup-args=-Davx2=disabled \
        --config-settings=setup-args=-Dsse2=disabled

You can also build against system-installed Blosc, Zstd, and LZ4 libraries
instead of the vendored copies::

    $ pip install -v --no-cache-dir --no-binary numcodecs numcodecs \
        --config-settings=setup-args=-Dsystem_blosc=enabled \
        --config-settings=setup-args=-Dsystem_zstd=enabled \
        --config-settings=setup-args=-Dsystem_lz4=enabled


To work with Numcodecs source code in development, see the
`contributing guide <contributing.html>`_ for instructions on setting up a
development environment with venv or uv.

```
$ git clone --recursive https://github.com/zarr-developers/numcodecs.git
$ cd numcodecs
$ pip install -e .[test,msgpack,zfpy]
```

Note: if you prefer to use the GitHub CLI `gh` you will need to append `-- --recurse-submodules`
to the clone command to everything works properly.

To verify that Numcodecs has been fully installed (including the Blosc
extension) run the test suite:

```
$ pytest -v
```

## Contents

```{toctree}
:maxdepth: 2

api
release
contributing
```

## Acknowledgments

The following people have contributed to the development of NumCodecs by contributing code,
documentation, code reviews, comments and/or ideas:

* {user}`Francesc Alted <FrancescAlted>`
* {user}`Prakhar Goel <newt0311>`
* {user}`Jerome Kelleher <jeromekelleher>`
* {user}`John Kirkham <jakirkham>`
* {user}`Alistair Miles <alimanfoo>`
* {user}`Jeff Reback <jreback>`
* {user}`Trevor Manz <manzt>`
* {user}`Grzegorz Bokota <Czaki>`
* {user}`Josh Moore <joshmoore>`
* {user}`Martin Durant <martindurant>`
* {user}`Paul Branson <pbranson>`

Numcodecs bundles the [c-blosc](https://github.com/Blosc/c-blosc) library.

Development of this package is supported by the
[MRC Centre for Genomics and Global Health](https://www.sanger.ac.uk/collaboration/mrc-centre-genomics-and-global-health-cggh/).

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
