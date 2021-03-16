Contributing to NumCodecs
=========================

NumCodecs is a community maintained project. We welcome contributions in the form of bug
reports, bug fixes, documentation, enhancement proposals and more. This page provides
information on how best to contribute.

Asking for help
---------------

If you have a question about how to use NumCodecs, please post your question on
StackOverflow using the `"numcodecs" tag <https://stackoverflow.com/questions/tagged/numcodecs>`_.
If you don't get a response within a day or two, feel free to raise a `GitHub issue
<https://github.com/zarr-developers/numcodecs/issues/new>`_ including a link to your
StackOverflow question. We will try to respond to questions as quickly as possible, but
please bear in mind that there may be periods where we have limited time to answer
questions due to other commitments.

Bug reports
-----------

If you find a bug, please raise a `GitHub issue
<https://github.com/zarr-developers/numcodecs/issues/new>`_. Please include the following items in
a bug report:

1. A minimal, self-contained snippet of Python code reproducing the problem. You can
   format the code nicely using markdown, e.g.::


    ```python
    >>> import numcodecs
    >>> codec = numcodecs.Zlib(1)
    ...
    ```

2. Information about the version of NumCodecs, along with versions of dependencies and the
   Python interpreter, and installation information. The version of NumCodecs can be obtained
   from the ``numcodecs.__version__`` property. Please also state how NumCodecs was installed,
   e.g., "installed via pip into a virtual environment", or "installed using conda".
   Information about other packages installed can be obtained by executing ``pip list``
   (if using pip to install packages) or ``conda list`` (if using conda to install
   packages) from the operating system command prompt. The version of the Python
   interpreter can be obtained by running a Python interactive session, e.g.::

    $ python
    Python 3.6.1 (default, Mar 22 2017, 06:17:05)
    [GCC 6.3.0 20170321] on linux

3. An explanation of why the current behaviour is wrong/not desired, and what you
   expect instead.

Enhancement proposals
---------------------

If you have an idea about a new feature or some other improvement to NumCodecs, please raise a
`GitHub issue <https://github.com/zarr-developers/numcodecs/issues/new>`_ first to discuss.

We very much welcome ideas and suggestions for how to improve NumCodecs, but please bear in
mind that we are likely to be conservative in accepting proposals for new features. The
reasons for this are that we would like to keep the NumCodecs code base lean and focused on
a core set of functionalities, and available time for development, review and maintenance
of new features is limited. But if you have a great idea, please don't let that stop
you posting it on GitHub, just please don't be offended if we respond cautiously.

Contributing code and/or documentation
--------------------------------------

Forking the repository
~~~~~~~~~~~~~~~~~~~~~~

The NumCodecs source code is hosted on GitHub at the following location:

* `https://github.com/zarr-developers/numcodecs <https://github.com/zarr-developers/numcodecs>`_

You will need your own fork to work on the code. Go to the link above and hit
the "Fork" button. Then clone your fork to your local machine::

    $ git clone git@github.com:your-user-name/numcodecs.git
    $ cd numcodecs
    $ git remote add upstream git@github.com:zarr-developers/numcodecs.git

Creating a development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To work with the NumCodecs source code, it is recommended to set up a Python virtual
environment and install all NumCodecs dependencies using the same versions as are used by
the core developers and continuous integration services. Assuming you have a Python
3 interpreter already installed, and have also installed the virtualenv package, and
you have cloned the NumCodecs source code and your current working directory is the root of
the repository, you can do something like the following::

    $ mkdir -p ~/pyenv/numcodecs-dev
    $ virtualenv --no-site-packages --python=/usr/bin/python3.9 ~/pyenv/numcodecs-dev
    $ source ~/pyenv/numcodecs-dev/bin/activate
    $ pip install -r requirements_dev.txt
    $ python setup.py build_ext --inplace

To verify that your development environment is working, you can run the unit tests::

    $ pytest -v numcodecs

Creating a branch
~~~~~~~~~~~~~~~~~

Before you do any new work or submit a pull request, please open an issue on GitHub to
report the bug or propose the feature you'd like to add.

It's best to create a new, separate branch for each piece of work you want to do. E.g.::

    git fetch upstream
    git checkout -b shiny-new-feature upsteam/master

This changes your working directory to the 'shiny-new-feature' branch. Keep any changes in
this branch specific to one bug or feature so it is clear what the branch brings to
NumCodecs.

To update this branch with latest code from NumCodecs, you can retrieve the changes from
the master branch and perform a rebase::

    git fetch upstream
    git rebase upstream/master

This will replay your commits on top of the latest NumCodecs git master. If this leads to
merge conflicts, these need to be resolved before submitting a pull request.
Alternatively, you can merge the changes in from upstream/master instead of rebasing,
which can be simpler::

    git fetch upstream
    git merge upstream/master

Again, any conflicts need to be resolved before submitting a pull request.

Running the test suite
~~~~~~~~~~~~~~~~~~~~~~

NumCodecs includes a suite of unit tests, as well as doctests included in function and class
docstrings. The simplest way to run the unit tests is to invoke::

    $ pytest -v numcodecs

To also run the doctests within docstrings, run::

    $ pytest -v --doctest-modules numcodecs

Tests can be run under different Python versions using tox. E.g. (assuming you have the
corresponding Python interpreters installed on your system)::

    $ tox -e py36,py37,py38,py39

NumCodecs currently supports Python 6-3.9, so the above command must
succeed before code can be accepted into the main code base. Note that only the py39
tox environment runs the doctests, i.e., doctests only need to succeed under Python 3.9.

All tests are automatically run via Travis (Linux) and AppVeyor (Windows) continuous
integration services for every pull request. Tests must pass under both services before
code can be accepted.

Code standards
~~~~~~~~~~~~~~

All code must conform to the PEP8 standard. Regarding line length, lines up to 100
characters are allowed, although please try to keep under 90 wherever possible.
Conformance can be checked by running::

    $ flake8 --max-line-length=100 numcodecs

This is automatically run when invoking ``tox -e py39``.

Test coverage
~~~~~~~~~~~~~

NumCodecs maintains 100% test coverage under the latest Python stable release (currently
Python 3.9). Both unit tests and docstring doctests are included when computing
coverage. Running ``tox -e py39`` will automatically run the test suite with coverage
and produce a coverage report. This should be 100% before code can be accepted into the
main code base.

When submitting a pull request, coverage will also be collected across all supported
Python versions via the Coveralls service, and will be reported back within the pull
request. Coveralls coverage must also be 100% before code can be accepted.

Documentation
~~~~~~~~~~~~~

Docstrings for user-facing classes and functions should follow the `numpydoc
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_ standard,
including sections for Parameters and Examples. All examples will be run as doctests
under Python 3.9.

NumCodecs uses Sphinx for documentation, hosted on readthedocs.org. Documentation is
written in the RestructuredText markup language (.rst files) in the ``docs`` folder.
The documentation consists both of prose and API documentation. All user-facing classes
and functions should be included in the API documentation. Any changes should also be
included in the release notes (``docs/release.rst``).

The documentation can be built by running::

    $ tox -e docs

The resulting built documentation will be available in the ``.tox/docs/tmp/html`` folder.

Development best practices, policies and procedures
---------------------------------------------------

The following information is mainly for core developers, but may also be of interest to
contributors.

Merging pull requests
~~~~~~~~~~~~~~~~~~~~~

Pull requests submitted by an external contributor should be reviewed and approved by at least
one core developers before being merged. Ideally, pull requests submitted by a core developer
should be reviewed and approved by at least one other core developers before being merged.

Pull requests should not be merged until all CI checks have passed (Travis, AppVeyor,
Coveralls) against code that has had the latest master merged in.

Compatibility and versioning policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because NumCodecs is a data encoding/decoding library, there are two types of compatibility to
consider: API compatibility and data format compatibility.

API compatibility
"""""""""""""""""

All functions, classes and methods that are included in the API
documentation (files under ``docs/api/*.rst``) are considered as part of the NumCodecs
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

Data format compatibility
"""""""""""""""""""""""""

Each codec class in NumCodecs exposes a ``codec_id`` attribute, which is an identifier for the
**format of the encoded data** produced by that codec. Thus it is valid for two or more codec
classes to expose the same value for the ``codec_id`` attribute if the format of the encoded data
is identical. The ``codec_id`` is intended to provide a basis for achieving and managing
interoperability between versions of the NumCodecs package, as well as between NumCodecs and other
software libraries that aim to provide compatible codec implementations. Currently there is no
formal specification of the encoded data format corresponding to each ``codec_id``, so the codec
classes provided in the NumCodecs package should be taken as the reference implementation for a
given ``codec_id``.

There must be a one-to-one mapping from ``codec_id`` values to encoded data formats, and that
mapping must not change once the first implementation of a ``codec_id`` has been published within a
NumCodecs release. If a change is proposed to the encoded data format for a particular type of
codec, then this must be implemented in NumCodecs via a new codec class exposing a new ``codec_id``
value.

Note that the NumCodecs test suite includes a data fixture and tests to try and ensure that
data format compatibility is not accidentally broken. See the
:func:`test_backwards_compatibility` functions in test modules for each codec for examples.

When to make a release
~~~~~~~~~~~~~~~~~~~~~~

Ideally, any bug fixes that don't change the public API should be released as soon as
possible. It is fine for a micro release to contain only a single bug fix.

When to make a minor release is at the discretion of the core developers. There are no
hard-and-fast rules, e.g., it is fine to make a minor release to make a single new
feature available; equally, it is fine to make a minor release that includes a number of
changes.

Major releases obviously need to be given careful consideration, and should be done as
infrequently as possible, as they will break existing code and/or affect data
compatibility in some way.

Release procedure
~~~~~~~~~~~~~~~~~

Checkout and update the master branch::

    $ git checkout master
    $ git pull

Verify all tests pass on all supported Python versions, and docs build::

    $ tox

Tag the version (where "X.X.X" stands for the version number, e.g., "2.2.0")::

    $ version=X.X.X
    $ git tag -a v$version -m v$version
    $ git push --tags

This will trigger a GitHub Action which will build the source
distribution as well as wheels for all major platforms.
