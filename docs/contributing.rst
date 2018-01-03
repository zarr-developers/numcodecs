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
    $ virtualenv --no-site-packages --python=/usr/bin/python3.6 ~/pyenv/numcodecs-dev
    $ source ~/pyenv/numcodecs-dev/bin/activate
    $ pip install -r requirements_dev.txt
    $ python setup.py build_ext --inplace

To verify that your development environment is working, you can run the unit tests::

    $ pytest -v numcodecs

Creating a branch
~~~~~~~~~~~~~~~~~

It's best to create a new, separate branch for each piece of work you want to do. E.g.::

    git checkout -b shiny-new-feature

This changes your working directory to the shiny-new-feature branch. Keep any changes in
this branch specific to one bug or feature so it is clear what the branch brings to
NumCodecs.

To update this branch with latest code from NumCodecs, you can retrieve the changes from
the master branch::

    git fetch upstream
    git rebase upstream/master

This will replay your commits on top of the latest NumCodecs git master. If this leads to
merge conflicts, these need to be resolved before submitting a pull request.

Before you do any new work or submit a pull request, please open an issue on GitHub to
report the bug or propose the feature you'd like to add.

Running the test suite
~~~~~~~~~~~~~~~~~~~~~~

NumCodecs includes a suite of unit tests, as well as doctests included in function and class
docstrings. The simplest way to run the unit tests is to invoke::

    $ pytest -v numcodecs

To also run the doctests within docstrings, run::

    $ pytest -v --doctest-modules numcodecs

Tests can be run under different Python versions using tox. E.g. (assuming you have the
corresponding Python interpreters installed on your system)::

    $ tox -e py27,py34,py35,py36

NumCodecs currently supports Python 2.7 and Python 3.4-3.6, so the above command must
succeed before code can be accepted into the main code base. Note that only the py36
tox environment runs the doctests, i.e., doctests only need to succeed under Python 3.6.

All tests are automatically run via Travis (Linux) and AppVeyor (Windows) continuous
integration services for every pull request. Tests must pass under both services before
code can be accepted.

Code standards
~~~~~~~~~~~~~~

All code must conform to the PEP8 standard. Regarding line length, lines up to 100
characters are allowed, although please try to keep under 90 wherever possible.
Conformance can be checked by running::

    $ flake8 --max-line-length=100 numcodecs

This is automatically run when invoking ``tox -e py36``.

Test coverage
~~~~~~~~~~~~~

NumCodecs maintains 100% test coverage under the latest Python stable release (currently
Python 3.6). Both unit tests and docstring doctests are included when computing
coverage. Running ``tox -e py36`` will automatically run the test suite with coverage
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
under Python 3.6.

NumCodecs uses Sphinx for documentation, hosted on readthedocs.org. Documentation is
written in the RestructuredText markup language (.rst files) in the ``docs`` folder.
The documentation consists both of prose and API documentation. All user-facing classes
and functions should be included in the API documentation. Any changes should also be
included in the release notes (``docs/release.rst``).

The documentation can be built by running::

    $ tox -e docs

The resulting built documentation will be available in the ``.tox/docs/tmp/html`` folder.
