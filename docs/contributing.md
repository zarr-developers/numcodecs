# Contributing

NumCodecs is a community maintained project. We welcome contributions in the form of bug
reports, bug fixes, documentation, enhancement proposals and more. This page provides
information on how best to contribute.

## Asking for help

If you have a question about how to use NumCodecs, please post your question on
StackOverflow using the ["numcodecs" tag](https://stackoverflow.com/questions/tagged/numcodecs).
If you don't get a response within a day or two, feel free to raise a [GitHub issue](https://github.com/zarr-developers/numcodecs/issues/new) including a link to your
StackOverflow question. We will try to respond to questions as quickly as possible, but
please bear in mind that there may be periods where we have limited time to answer
questions due to other commitments.

## Bug reports

If you find a bug, please raise a [GitHub issue](https://github.com/zarr-developers/numcodecs/issues/new). Please include the following items in
a bug report:

1. A minimal, self-contained snippet of Python code reproducing the problem. You can
   format the code nicely using markdown, e.g.:

    ```python
    >>> import numcodecs
    >>> codec = numcodecs.Zlib(1)
    ...
    ```

2. Information about the version of NumCodecs, along with versions of dependencies and the
   Python interpreter, and installation information. The version of NumCodecs can be obtained
   from the `numcodecs.__version__` property. Please also state how NumCodecs was installed,
   e.g., "installed via pip into a virtual environment", or "installed using conda".
   Information about other packages installed can be obtained by executing ``pip list``
   (if using pip to install packages) or ``conda list`` (if using conda to install
   packages) from the operating system command prompt.

3. An explanation of why the current behaviour is wrong/not desired, and what you
   expect instead.

## Enhancement proposals

If you have an idea about a new feature or some other improvement to NumCodecs, please raise a
[GitHub issue](https://github.com/zarr-developers/numcodecs/issues/new) first to discuss.

We very much welcome ideas and suggestions for how to improve NumCodecs, but please bear in
mind that we are likely to be conservative in accepting proposals for new features. The
reasons for this are that we would like to keep the NumCodecs code base lean and focused on
a core set of functionalities, and available time for development, review and maintenance
of new features is limited. But if you have a great idea, please don't let that stop
you posting it on GitHub, just please don't be offended if we respond cautiously.

## Contributing code and/or documentation

### Forking the repository

The NumCodecs source code is hosted on GitHub at the following location:

* <https://github.com/zarr-developers/numcodecs>

You will need your own fork to work on the code. Go to the link above and hit
the "Fork" button. Then clone your fork to your local machine:

```
$ git clone --recursive git@github.com:your-user-name/numcodecs.git  # with ssh
```

or:

```
$ git clone --recursive https://github.com/your-user-name/numcodecs.git  # with https
```

Then `cd` into the clone and add the `upstream` remote:

```
$ cd numcodecs
$ git remote add upstream https://github.com/zarr-developers/numcodecs.git
```

Note the ``--recursive`` flag is required to clone the ``c-blosc`` git submodule. If you
forgot it, you can initialize it later with:

```
$ git submodule update --init --recursive
```

### Creating a development environment

NumCodecs contains C and Cython extensions, so you need a C compiler and build
tooling in addition to Python.

#### Prerequisites

You need a C compiler available on your ``PATH``.

On Debian/Ubuntu:

```
$ sudo apt install build-essential
```

On macOS (Xcode command line tools):

```
$ xcode-select --install
```

On Windows, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) with the "Desktop development
with C++" workload.

#### Setting up with uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. First,
bootstrap the build dependencies, then install numcodecs in editable mode:

```
$ uv venv
$ uv pip install --group dev
$ uv pip install --no-build-isolation -e ".[test,test_extras,msgpack]"
```

The first ``uv pip install`` step bootstraps the build tools into the virtualenv.
The second installs numcodecs in editable mode without build isolation (so the
venv's build tools are used). This two-step process is needed because
meson-python editable installs require build tools at runtime for auto-rebuild
on import.

To run the tests:

```
$ uv run pytest -v
```

#### Setting up with venv and pip

If you prefer the standard library ``venv``:

```
$ python -m venv venv
$ source venv/bin/activate        # macOS/Linux
$ # .\venv\Scripts\activate       # Windows

$ pip install --group dev
$ pip install --no-build-isolation -e ".[test,test_extras,msgpack]"
$ pytest -v
```

#### Passing build options

NumCodecs uses [Meson](https://mesonbuild.com) as its build system. You can
pass Meson options via pip's ``--config-settings``.

To build against system-installed libraries instead of the vendored copies:

```
$ pip install --no-build-isolation -e . \
    --config-settings=setup-args=-Dsystem_blosc=enabled \
    --config-settings=setup-args=-Dsystem_zstd=enabled \
    --config-settings=setup-args=-Dsystem_lz4=enabled
```

To disable SIMD optimizations (e.g., for portable debugging):

```
$ pip install --no-build-isolation -e . \
    --config-settings=setup-args=-Davx2=disabled \
    --config-settings=setup-args=-Dsse2=disabled
```

#### Rebuilding after changes

With an editable install, meson-python automatically rebuilds changed C/Cython
extensions when you import ``numcodecs``. If you need a full clean rebuild:

```
$ rm -rf build
$ pip install --no-build-isolation -e .
```

### Creating a branch

Before you do any new work or submit a pull request, please open an issue on GitHub to
report the bug or propose the feature you'd like to add.

It's best to create a new, separate branch for each piece of work you want to do. E.g.:

```
git fetch upstream
git checkout -b shiny-new-feature upstream/main
```

This changes your working directory to the 'shiny-new-feature' branch. Keep any changes in
this branch specific to one bug or feature so it is clear what the branch brings to
NumCodecs.

To update this branch with latest code from NumCodecs, you can retrieve the changes from
the main branch and perform a rebase:

```
git fetch upstream
git rebase upstream/main
```

This will replay your commits on top of the latest NumCodecs git main. If this leads to
merge conflicts, these need to be resolved before submitting a pull request.
Alternatively, you can merge the changes in from upstream/main instead of rebasing,
which can be simpler:

```
git fetch upstream
git merge upstream/main
```

Again, any conflicts need to be resolved before submitting a pull request.

### Running the test suite

NumCodecs includes a suite of unit tests, as well as doctests included in function and class
docstrings:

```
$ uv run pytest -v
```

Or, if using venv/pip:

```
$ pytest -v
```

To test against specific Zarr-Python versions, use the dependency groups defined in
``pyproject.toml``:

```
$ uv run --group test-zarr-312 pytest tests/test_zarr3.py tests/test_zarr3_import.py
$ uv run --group test-zarr-313 pytest tests/test_zarr3.py tests/test_zarr3_import.py
$ uv run --group test-zarr-main pytest tests/test_zarr3.py tests/test_zarr3_import.py
```

Or, if using venv/pip:

```
$ pip install --group test-zarr-312
$ pytest tests/test_zarr3.py tests/test_zarr3_import.py
```

To test with different CRC32C implementations:

```
$ uv pip install "numcodecs[crc32c]"        # sw-based crc32c
$ pytest tests/test_checksum32.py -v

$ uv pip install "numcodecs[google_crc32c]"  # hw-accelerated google-crc32c
$ pytest tests/test_checksum32.py -v
```

All tests are automatically run via GitHub Actions for every pull request across Linux
(x86_64, aarch64, i386), macOS (x86_64, arm64), and Windows (x86_64), with Python 3.12
through 3.14. Tests must pass on all platforms before code can be accepted.

### Code standards

All code must conform to the PEP8 standard. Regarding line length, lines up to 100
characters are allowed, although please try to keep under 90 wherever possible.
Conformance can be checked by running:

```
$ pre-commit run ruff
```

### Test coverage

NumCodecs maintains 100% test coverage under the latest stable Python release. Both unit
tests and docstring doctests are included when computing coverage. Running ``pytest -v``
will automatically run the test suite with coverage and produce a coverage report. This
should be 100% before code can be accepted into the main code base.

When submitting a pull request, coverage will also be collected across all supported
Python versions via the Codecov service, and will be reported back within the pull
request. Codecov coverage must also be 100% before code can be accepted.

### Documentation

Docstrings for user-facing classes and functions should follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) standard,
including sections for Parameters and Examples. All examples will be run as doctests.

NumCodecs uses Sphinx for documentation, hosted on readthedocs.org. Documentation is
written in Markdown (.md files) using [MyST](https://myst-parser.readthedocs.io/) in the `docs` folder.
The documentation consists both of prose and API documentation. All user-facing classes
and functions should be included in the API documentation. Any changes should also be
included in the release notes (`docs/release.md`).

The documentation can be built by running:

```
$ cd docs
$ make clean; make html
```

The resulting built documentation will be available in the `docs/_build/html` folder.

## Development best practices, policies and procedures

The following information is mainly for core developers, but may also be of interest to
contributors.

### Merging pull requests

Pull requests should be reviewed and approved by at least one core developer
(other than the pull request author) before being merged.

Pull requests should not be merged until all CI checks have passed against code
that has had the latest main merged in.

### Compatibility and versioning policies

Because NumCodecs is a data encoding/decoding library, there are two types of compatibility to
consider: API compatibility and data format compatibility.

#### API compatibility

All functions, classes and methods that are included in the API
documentation (files under `docs/api/*.md`) are considered as part of the NumCodecs
**public API**, except if they have been documented as an experimental feature, in which case they
are part of the **experimental API**.

Any change to the public API that does **not** break existing third party
code importing NumCodecs, or cause third party code to behave in a different way, is a
**backwards-compatible API change**. For example, adding a new function, class or method is usually
a backwards-compatible change. However, removing a function, class or method; removing an argument
to a function or method; adding a required argument to a function or method; or changing the
behaviour of a function or method, are examples of **backwards-incompatible API changes**.

If a release contains no changes to the public API (e.g., contains only bug fixes or
other maintenance work), then the micro version number should be incremented (e.g.,
2.2.0 -> 2.2.1). If a release contains public API changes, but all changes are
backwards-compatible, then the minor version number should be incremented
(e.g., 2.2.1 -> 2.3.0). If a release contains any backwards-incompatible public API changes,
the major version number should be incremented (e.g., 2.3.0 -> 3.0.0).

Backwards-incompatible changes to the experimental API can be included in a minor release,
although this should be minimised if possible. I.e., it would be preferable to save up
backwards-incompatible changes to the experimental API to be included in a major release, and to
stabilise those features at the same time (i.e., move from experimental to public API), rather than
frequently tinkering with the experimental API in minor releases.

#### Data format compatibility

Each codec class in NumCodecs exposes a `codec_id` attribute, which is an identifier for the
**format of the encoded data** produced by that codec. Thus it is valid for two or more codec
classes to expose the same value for the `codec_id` attribute if the format of the encoded data
is identical. The `codec_id` is intended to provide a basis for achieving and managing
interoperability between versions of the NumCodecs package, as well as between NumCodecs and other
software libraries that aim to provide compatible codec implementations. Currently there is no
formal specification of the encoded data format corresponding to each `codec_id`, so the codec
classes provided in the NumCodecs package should be taken as the reference implementation for a
given `codec_id`.

There must be a one-to-one mapping from `codec_id` values to encoded data formats, and that
mapping must not change once the first implementation of a `codec_id` has been published within a
NumCodecs release. If a change is proposed to the encoded data format for a particular type of
codec, then this must be implemented in NumCodecs via a new codec class exposing a new `codec_id`
value.

Note that the NumCodecs test suite includes a data fixture and tests to try and ensure that
data format compatibility is not accidentally broken. See the
{func}`test_backwards_compatibility` functions in test modules for each codec for examples.

### When to make a release

Ideally, any bug fixes that don't change the public API should be released as soon as
possible. It is fine for a micro release to contain only a single bug fix.

When to make a minor release is at the discretion of the core developers. There are no
hard-and-fast rules, e.g., it is fine to make a minor release to make a single new
feature available; equally, it is fine to make a minor release that includes a number of
changes.

When making a minor release, open an issue stating your intention so other developers
know that a release is planned. At least a week's notice should be given for other
developers to be aware of and possibly add to the contents of the release.

Major releases obviously need to be given careful consideration, and should be done as
infrequently as possible, as they will break existing code and/or affect data
compatibility in some way.

### Release procedure

Checkout and update the main branch:

```
$ git checkout main
$ git pull
```

Tag the version (where "X.X.X" stands for the version number, e.g., "2.2.0"):

```
$ version=X.X.X
$ git tag -a v$version -m v$version
$ git push --tags
```

This will trigger a GitHub Action which will build the source
distribution as well as wheels for all major platforms.
