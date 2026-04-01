# Design Doc: Migrate numcodecs build system from setuptools/setup.py to meson-python

**Status:** Proposal
**Date:** 2026-03-17
**Author:** Max (with Claude)

## Motivation

numcodecs currently uses a hybrid build system: `pyproject.toml` declares the setuptools backend, but a 386-line `setup.py` contains all the C/Cython extension logic. This is problematic because:

1. **`setup.py` is effectively deprecated.** PEP 517/518 moved the ecosystem toward declarative builds. `setup.py` is imperative, opaque to tooling, and executed during metadata extraction.
2. **`distutils` imports are fragile.** `setup.py` imports from `distutils` (removed from stdlib in Python 3.12). This only works because setuptools vendors a copy, which may be dropped in a future major release.
3. **CPU feature detection is incorrect for cross-compilation.** `py-cpuinfo` runs at build time on the *build* machine, not the *target* machine. This means cibuildwheel arm64 builds on x86_64 hosts detect x86_64 features. The current workaround is environment variables (`DISABLE_NUMCODECS_AVX2`, `DISABLE_NUMCODECS_SSE2`).
4. **System library linking is unsupported.** PR [#569](https://github.com/zarr-developers/numcodecs/pull/569) has been open since 2023, attempting to add `pkg-config`-based system library support. The approach adds complexity to an already complex `setup.py`.
5. **The scientific Python ecosystem has standardized on meson-python.** numpy, scipy, scikit-learn, scikit-image, and matplotlib have all migrated. Meson natively handles every concern currently implemented manually in `setup.py`.

## Proposal

Replace `setup.py` + setuptools with `meson.build` + `meson-python` as the build backend. Remove `py-cpuinfo` as a build dependency. Use meson's native compiler introspection for CPU feature detection and `dependency()` for optional system library linking.

## Current Architecture

### Build dependencies

```toml
[build-system]
requires = ["setuptools>=77", "setuptools-scm[toml]>=6.2", "Cython", "py-cpuinfo", "numpy>2"]
build-backend = "setuptools.build_meta"
```

### Extension modules (8 total)

| Extension | Cython source | C library deps | Assembly | Notes |
|-----------|--------------|----------------|----------|-------|
| `blosc` | `blosc.pyx` | c-blosc + vendored lz4, zlib, zstd | zstd `*amd64.S` / `*aarch64.S` | Most complex; SSE2/AVX2 conditional |
| `zstd` | `zstd.pyx` | vendored zstd (from c-blosc) | zstd `*amd64.S` / `*aarch64.S` | Shares zstd sources with blosc |
| `lz4` | `lz4.pyx` | vendored lz4 (from c-blosc) | None | |
| `vlen` | `vlen.pyx` | None | None | Needs numpy include dirs |
| `fletcher32` | `fletcher32.pyx` | None | None | |
| `jenkins` | `jenkins.pyx` | None | None | Has `CYTHON_TRACE=1` (likely accidental) |
| `compat_ext` | `compat_ext.pyx` | None | None | |
| `_shuffle` | `_shuffle.pyx` | None | None | |

### Shared Cython declarations

- `compat_ext.pxd` — `PyBytes_RESIZE`, `ensure_continguous_memoryview` (used by blosc, zstd, lz4, vlen)
- `_utils.pxd` — `store_le32`, `load_le32` (used by lz4, fletcher32, vlen)

### Vendored C sources (c-blosc git submodule)

```
c-blosc/
├── blosc/              # Core blosc library (28 files, incl. SIMD shuffle variants)
└── internal-complibs/
    ├── lz4-1.10.0/     # 4 files
    ├── zlib-1.3.1/     # ~58 files
    └── zstd-1.5.6/     # ~90 files across common/, compress/, decompress/, dictBuilder/
```

### Conditional compilation logic in setup.py

1. **CPU detection:** `py-cpuinfo` reads `/proc/cpuinfo` or equivalent → sets `have_sse2`, `have_avx2`
2. **Environment overrides:** `DISABLE_NUMCODECS_SSE2`, `DISABLE_NUMCODECS_AVX2`, `DISABLE_NUMCODECS_CEXT`
3. **SIMD sources:** SSE2/AVX2 shuffle/bitshuffle `.c` files included conditionally
4. **SIMD defines:** `-DSHUFFLE_SSE2_ENABLED`, `-DSHUFFLE_AVX2_ENABLED`, `__SSE2__`, `__AVX2__`
5. **Assembly:** `.S` files pre-compiled to `.o`, passed as `extra_objects`
6. **Platform flags:** `-stdlib=libc++` (macOS), `-pthread` (POSIX), `-Wno-implicit-function-declaration` (cibuildwheel macOS)

## Proposed Architecture

### Build dependencies

```toml
[build-system]
requires = ["meson-python>=0.17", "meson>=1.6.0", "Cython>=3.0", "numpy>=2"]
build-backend = "mesonpy"
```

**Removed:** `setuptools`, `setuptools-scm`, `py-cpuinfo`

### File structure

```
numcodecs/
├── meson.build                    # Root: project(), subdir()
├── meson.options                  # Build options (system libs, SIMD toggles)
├── numcodecs/
│   └── meson.build                # Extension modules
├── c-blosc/                       # Unchanged submodule
├── pyproject.toml                 # Updated build-backend
└── setup.py                       # DELETED
```

### Root meson.build

```meson
project(
  'numcodecs',
  'c', 'cython',
  version: run_command('python', '-m', 'setuptools_scm', '--version', check: true).stdout().strip(),
  default_options: [
    'c_std=c11',
    'warning_level=1',
  ],
  meson_version: '>=1.6.0',
)

py = import('python').find_installation(pure: false)
cy = meson.get_compiler('cython')
cc = meson.get_compiler('c')

# NumPy include directory (needed for vlen extension)
numpy_dep = dependency('numpy')

subdir('numcodecs')
```

**Note on versioning:** meson-python has native setuptools-scm integration — see [meson-python docs on dynamic versioning](https://mesonbuild.com/meson-python/how-to-guides/dynamic-version.html). The `version` field can use `run_command` to call setuptools-scm, or we can switch to `meson-python`'s built-in VCS versioning. This will need to replace the current `write_to = "numcodecs/version.py"` approach — meson-python generates version metadata at build time via importlib.metadata instead of writing a file.

### meson.options

```meson
option('system_blosc', type: 'feature', value: 'auto',
       description: 'Use system-installed Blosc library')
option('system_zstd', type: 'feature', value: 'auto',
       description: 'Use system-installed Zstandard library')
option('system_lz4', type: 'feature', value: 'auto',
       description: 'Use system-installed LZ4 library')
```

With `value: 'disabled'`, meson always uses the vendored sources by default, matching the old setup.py behavior. Users who want to link against system libraries opt in with `-Dsystem_blosc=enabled` etc. This replaces the `NUMCODECS_USE_SYSTEM_LIBS` env var from PR #569 with a standard meson idiom.

### numcodecs/meson.build (core of the migration)

This is the most complex file. The design below is broken into sections.

#### Compiler flags

```meson
# Platform-specific flags
c_args = []
link_args = []

if host_machine.system() == 'darwin'
  c_args += ['-stdlib=libc++']
endif

if host_machine.system() != 'windows'
  c_args += ['-pthread']
  link_args += ['-pthread']
endif
```

#### SIMD detection (replaces py-cpuinfo)

```meson
# SIMD support — detected via compiler capability, NOT runtime CPU flags.
# This is correct for cross-compilation: we check what the *target* supports.
have_sse2 = false
have_avx2 = false

if host_machine.cpu_family() == 'x86_64'
  have_sse2 = cc.has_argument('-msse2')
  have_avx2 = cc.has_argument('-mavx2')
endif

# Allow disabling via meson configure (replaces DISABLE_NUMCODECS_* env vars)
# cibuildwheel can pass -Dsse2=disabled instead of DISABLE_NUMCODECS_SSE2=1
```

This fixes the cross-compilation bug: `host_machine.cpu_family()` reflects the *target* architecture (from a meson cross file or native detection), not the build machine.

#### Vendored libraries as static dependencies

```meson
# --- Vendored zstd ---
zstd_dep = dependency('libzstd', required: get_option('system_zstd'))

if not zstd_dep.found()
  zstd_sources = files(
    # common/
    'c-blosc/internal-complibs/zstd-1.5.6/common/entropy_common.c',
    'c-blosc/internal-complibs/zstd-1.5.6/common/error_private.c',
    # ... (enumerate all .c files)
  )

  # Assembly files — meson handles .S natively
  if host_machine.cpu_family() == 'x86_64'
    zstd_sources += files('c-blosc/internal-complibs/zstd-1.5.6/decompress/huf_decompress_amd64.S')
  elif host_machine.cpu_family() == 'aarch64'
    zstd_sources += files('c-blosc/internal-complibs/zstd-1.5.6/decompress/huf_decompress_aarch64.S')
  endif

  zstd_inc = include_directories(
    'c-blosc/internal-complibs/zstd-1.5.6',
    'c-blosc/internal-complibs/zstd-1.5.6/common',
    'c-blosc/internal-complibs/zstd-1.5.6/compress',
    'c-blosc/internal-complibs/zstd-1.5.6/decompress',
    'c-blosc/internal-complibs/zstd-1.5.6/dictBuilder',
  )

  zstd_lib = static_library('zstd_vendored', zstd_sources,
    include_directories: zstd_inc,
    c_args: c_args,
  )
  zstd_dep = declare_dependency(
    link_with: zstd_lib,
    include_directories: zstd_inc,
  )
endif

# --- Vendored lz4 ---
lz4_dep = dependency('liblz4', required: get_option('system_lz4'))

if not lz4_dep.found()
  lz4_sources = files(
    'c-blosc/internal-complibs/lz4-1.10.0/lz4.c',
    'c-blosc/internal-complibs/lz4-1.10.0/lz4hc.c',
  )
  lz4_inc = include_directories('c-blosc/internal-complibs/lz4-1.10.0')

  lz4_lib = static_library('lz4_vendored', lz4_sources,
    include_directories: lz4_inc,
    c_args: c_args,
  )
  lz4_dep = declare_dependency(
    link_with: lz4_lib,
    include_directories: lz4_inc,
  )
endif

# --- Vendored zlib ---
zlib_sources = files(
  'c-blosc/internal-complibs/zlib-1.3.1/adler32.c',
  'c-blosc/internal-complibs/zlib-1.3.1/crc32.c',
  'c-blosc/internal-complibs/zlib-1.3.1/deflate.c',
  'c-blosc/internal-complibs/zlib-1.3.1/inflate.c',
  'c-blosc/internal-complibs/zlib-1.3.1/inffast.c',
  'c-blosc/internal-complibs/zlib-1.3.1/inftrees.c',
  'c-blosc/internal-complibs/zlib-1.3.1/trees.c',
  'c-blosc/internal-complibs/zlib-1.3.1/uncompr.c',
  'c-blosc/internal-complibs/zlib-1.3.1/zutil.c',
)
zlib_inc = include_directories('c-blosc/internal-complibs/zlib-1.3.1')

zlib_lib = static_library('zlib_vendored', zlib_sources,
  include_directories: zlib_inc,
  c_args: c_args,
)
zlib_dep = declare_dependency(
  link_with: zlib_lib,
  include_directories: zlib_inc,
)
```

#### Blosc extension (most complex)

```meson
# --- Vendored blosc ---
blosc_dep = dependency('blosc', required: get_option('system_blosc'))

if not blosc_dep.found()
  blosc_sources = files(
    'c-blosc/blosc/blosc.c',
    'c-blosc/blosc/blosclz.c',
    'c-blosc/blosc/fastcopy.c',
    'c-blosc/blosc/shuffle.c',
    'c-blosc/blosc/shuffle-generic.c',
    'c-blosc/blosc/bitshuffle-generic.c',
  )

  blosc_c_args = c_args
  blosc_c_args += ['-DHAVE_LZ4', '-DHAVE_ZLIB', '-DHAVE_ZSTD']

  if have_sse2
    blosc_sources += files(
      'c-blosc/blosc/shuffle-sse2.c',
      'c-blosc/blosc/bitshuffle-sse2.c',
    )
    blosc_c_args += ['-DSHUFFLE_SSE2_ENABLED', '-msse2']
    if host_machine.system() == 'windows'
      blosc_c_args += ['-D__SSE2__']
    endif
  endif

  if have_avx2
    blosc_sources += files(
      'c-blosc/blosc/shuffle-avx2.c',
      'c-blosc/blosc/bitshuffle-avx2.c',
    )
    blosc_c_args += ['-DSHUFFLE_AVX2_ENABLED', '-mavx2']
    if host_machine.system() == 'windows'
      blosc_c_args += ['-D__AVX2__']
    endif
  endif

  blosc_inc = include_directories('c-blosc/blosc')

  blosc_lib = static_library('blosc_vendored', blosc_sources,
    include_directories: blosc_inc,
    dependencies: [lz4_dep, zlib_dep, zstd_dep],
    c_args: blosc_c_args,
  )
  blosc_dep = declare_dependency(
    link_with: blosc_lib,
    include_directories: blosc_inc,
  )
endif
```

#### Extension module declarations

```meson
# --- Cython extension modules ---

# Extensions with C library dependencies
py.extension_module('blosc',
  'blosc.pyx',
  dependencies: [blosc_dep, lz4_dep, zlib_dep, zstd_dep],
  link_args: link_args,
  install: true,
  subdir: 'numcodecs',
)

py.extension_module('zstd',
  'zstd.pyx',
  dependencies: [zstd_dep],
  link_args: link_args,
  install: true,
  subdir: 'numcodecs',
)

py.extension_module('lz4',
  'lz4.pyx',
  dependencies: [lz4_dep],
  include_directories: include_directories('.'),
  link_args: link_args,
  install: true,
  subdir: 'numcodecs',
)

py.extension_module('vlen',
  'vlen.pyx',
  dependencies: [numpy_dep],
  include_directories: include_directories('.'),
  install: true,
  subdir: 'numcodecs',
)

# Pure-Cython extensions (no C library deps)
foreach ext : ['fletcher32', 'jenkins', 'compat_ext', '_shuffle']
  py.extension_module(ext,
    ext + '.pyx',
    include_directories: include_directories('.'),
    install: true,
    subdir: 'numcodecs',
  )
endforeach

# --- Pure Python sources ---
py.install_sources(
  '__init__.py',
  'abc.py',
  'compat.py',
  'registry.py',
  # ... all .py files
  subdir: 'numcodecs',
)
```

### Version management

**Option A: meson-python + setuptools-scm (minimal change)**

Keep `setuptools-scm` as a build dependency and use `run_command()` in `meson.build`:

```meson
project('numcodecs', 'c', 'cython',
  version: run_command(
    py, ['-m', 'setuptools_scm', '--version'], check: true
  ).stdout().strip(),
)
```

At runtime, version is read via `importlib.metadata.version('numcodecs')` instead of a generated `version.py`. Update `__init__.py`:

```python
from importlib.metadata import version
__version__ = version("numcodecs")
```

**Option B: meson's built-in VCS tagging**

Use `vcs_tag()` to generate a version file from git tags, removing the setuptools-scm dependency entirely. This is what numpy does.

**Recommendation:** Option A — keep setuptools-scm for now, swap to `vcs_tag()` later if desired.

### pyproject.toml changes

```toml
[build-system]
requires = ["meson-python>=0.17", "meson>=1.6.0", "Cython>=3.0", "numpy>=2"]
build-backend = "mesonpy"

[project]
# ... unchanged ...

[tool.meson-python.args]
# Pass default meson options
setup = ['--default-library=static']

# Replace setuptools-scm config
[tool.setuptools_scm]  # REMOVE this section

# Replace setuptools config
[tool.setuptools]      # REMOVE this section
```

### cibuildwheel changes

The `DISABLE_NUMCODECS_*` environment variables are replaced by meson configure options:

```toml
[tool.cibuildwheel]
# Meson cross-compilation handles SIMD correctly — no env vars needed
# AVX2 still disabled for portable wheels (not all x86_64 CPUs have it)
config-settings = "setup-args=-Davx2=disabled"

[tool.cibuildwheel.macos]
# No need for MACOSX_DEPLOYMENT_TARGET — meson-python handles this
# No need for -Wno-implicit-function-declaration — meson compiles correctly

[[tool.cibuildwheel.overrides]]
select = "*-macosx_arm64"
config-settings = "setup-args=-Davx2=disabled -Dsse2=disabled"
```

### Files deleted

- `setup.py` (386 lines)
- `MANIFEST.in` (meson-python builds sdists from version control, like setuptools-scm)
- `numcodecs/version.py` (replaced by `importlib.metadata`)

### Files added

- `meson.build` (root, ~20 lines)
- `meson.options` (~15 lines)
- `numcodecs/meson.build` (~180 lines)

### Files updated

- `docs/contributing.rst` (rewritten — see [Contributing Guide Updates](#contributing-guide-updates) below)

## Migration Plan

numcodecs has limited maintenance bandwidth. A phased migration (the numpy/scipy approach of running two build systems in parallel for multiple releases) would double the maintenance surface during the transition. Instead, this should be a single PR that swaps everything at once.

### Single-PR migration

One PR, one release. The PR:

1. **Adds** `meson.build`, `meson.options`, `numcodecs/meson.build`
2. **Deletes** `setup.py`, `MANIFEST.in`
3. **Updates** `pyproject.toml`: new `[build-system]`, removes `[tool.setuptools]`, `[tool.setuptools_scm]`, `[tool.setuptools.package-data]`
4. **Updates** `numcodecs/__init__.py`: version from `importlib.metadata`
5. **Updates** cibuildwheel config in `pyproject.toml`
6. **Updates** CI workflows (pixi tasks, wheel building)
7. **Updates** `.gitignore` (remove `numcodecs/version.py`)
8. **Updates** `docs/contributing.rst`: new dev setup instructions for pixi and uv, removes outdated references
9. **Closes** PR #569 (system library support is native via `-Dsystem_blosc=enabled`)

### Validation checklist

Before merging, verify:

- [ ] `pip install .` works (non-editable)
- [ ] `pip install -e . --no-build-isolation` works (editable)
- [ ] `python -m build --sdist` produces correct sdist (includes c-blosc submodule)
- [ ] `python -m build --wheel` produces working wheel
- [ ] cibuildwheel produces wheels for all target platforms
- [ ] All tests pass on Linux x86_64, Linux aarch64, macOS arm64, macOS x86_64, Windows x86_64
- [ ] i386 Alpine CI still works (may need a meson cross file)
- [ ] `pixi run run-tests` works for local development
- [ ] `-Dsystem_blosc=enabled` links against system blosc (replaces PR #569)
- [ ] Version string is correct in built package (`python -c "import numcodecs; print(numcodecs.__version__)"`)

### Risk of a clean break

The main risk is that a broken build blocks the next release. This is mitigated by:

- **The test matrix is comprehensive.** CI already tests 5 OS/arch combos x 4 Python versions. If the meson build passes CI, it works.
- **meson-python is battle-tested.** numpy, scipy, scikit-learn all use it for projects with far more complex C/Cython builds than numcodecs.
- **Rollback is trivial.** If something goes wrong post-release, revert the PR and cut a patch release. The old `setup.py` is in git history.

## Contributing Guide Updates

The current `docs/contributing.rst` is significantly outdated (references Python 3.8/3.9, Travis CI, AppVeyor, bare `pip install -e .` without a venv manager). The meson migration is a natural point to rewrite the development setup section.

### Current problems with the contributing guide

- Recommends manual `python3 -m venv` + `pip install -e .` — works but requires the user to have a C compiler, Cython, and numpy already available, with no guidance on obtaining them
- No mention of pixi or uv, which are already configured in the project
- References Python 3.8, 3.9, Travis CI, and AppVeyor (all outdated)
- No mention of meson or how to pass build options
- No guidance on building against system libraries

### Proposed "Creating a development environment" section

The guide should present two supported paths (pixi and uv) and explicitly discourage developing without an isolated environment. The following replaces the "Creating a development environment" and "Running the test suite" sections.

```rst
Creating a development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NumCodecs contains C and Cython extensions, so you need a C compiler and
build tooling in addition to Python. We support two development workflows:
**pixi** (recommended) and **uv**. Both manage isolated environments and
are tested in CI. **Do not** install numcodecs into your system Python or a
manually-managed virtualenv — the C compilation dependencies are
difficult to get right by hand.

Whichever tool you use, first clone the repository with its submodules::

    $ git clone --recursive git@github.com:your-user-name/numcodecs.git
    $ cd numcodecs
    $ git remote add upstream https://github.com/zarr-developers/numcodecs.git

Using pixi (recommended)
"""""""""""""""""""""""""

`pixi <https://pixi.sh>`_ manages Python, the C compiler toolchain, and all
dependencies via conda-forge. This is the easiest way to get started,
especially on macOS or if you don't have a system C compiler.

Install pixi, then::

    $ pixi run -e test-py313 run-tests     # build + test in one step

This will create an isolated environment with Python 3.13, install the C
toolchain from conda-forge, build the Cython extensions, and run the test
suite. Available test environments::

    test-py311, test-py312, test-py313, test-py314

To get a shell inside a pixi environment for interactive development::

    $ pixi shell -e test-py313

Using uv
""""""""

`uv <https://docs.astral.sh/uv/>`_ is a fast Python package manager. It
manages virtualenvs and dependencies but does **not** provide a C compiler —
you must have ``cc`` / ``gcc`` / ``clang`` available on your ``PATH``.

On Debian/Ubuntu::

    $ sudo apt install build-essential

On macOS (Xcode command line tools)::

    $ xcode-select --install

Then::

    $ uv sync --extra test --extra msgpack
    $ uv run pytest -v

uv will create a ``.venv``, build the extensions, and install numcodecs in
editable mode.

Passing build options
"""""""""""""""""""""

NumCodecs uses `meson <https://mesonbuild.com>`_ as its build system. You can
pass meson options via pip's ``--config-settings`` or via environment variables
when building.

To build against system-installed Blosc, Zstd, and LZ4 instead of the
vendored copies::

    $ pip install -e . --no-build-isolation \
        --config-settings=setup-args=-Dsystem_blosc=enabled \
        --config-settings=setup-args=-Dsystem_zstd=enabled \
        --config-settings=setup-args=-Dsystem_lz4=enabled

To disable SIMD optimizations (e.g. for portable debugging)::

    $ pip install -e . --no-build-isolation \
        --config-settings=setup-args=-Davx2=disabled \
        --config-settings=setup-args=-Dsse2=disabled

Running the test suite
~~~~~~~~~~~~~~~~~~~~~~

With pixi::

    $ pixi run -e test-py313 run-tests

With uv::

    $ uv run pytest -v

To run tests for specific Zarr integration versions::

    $ pixi run test-zarr-312
    $ pixi run test-zarr-313
    $ pixi run test-zarr-main

All tests are automatically run via GitHub Actions for every pull request
across Linux (x86_64, aarch64, i386), macOS (x86_64, arm64), and Windows
(x86_64), with Python 3.11 through 3.14.
```

### Other sections to update

The rest of the contributing guide needs lighter touch-ups as part of this PR:

- **"Code standards"** — already correct (references `pre-commit run ruff`), no change needed
- **"Test coverage"** — update Python version reference from "3.9" to "3.13" (or just say "the latest stable release")
- **"Documentation"** — update Python version reference, otherwise fine
- **"Bug reports"** — update the Python session example from 3.8 to a current version
- **"Running the test suite" (old)** — replace entirely (covered above)
- Remove all references to Travis CI and AppVeyor — CI is GitHub Actions only

## Risks and Mitigations

### Risk: Meson doesn't support a platform/arch that setuptools did

**Mitigation:** Meson supports all platforms numcodecs targets (Linux x86_64/aarch64/i386, macOS x86_64/arm64, Windows x86_64). The i386 Alpine CI job (`ci-i386.yml`) needs verification — meson may need a cross file for 32-bit builds inside the Alpine container.

### Risk: Cython version incompatibilities

**Mitigation:** Pin `Cython>=3.0` in build-requires. Meson's Cython support requires Cython 3. This drops Cython 0.x support, which is already effectively unsupported.

### Risk: sdist contents change

**Mitigation:** meson-python includes all files tracked by git (plus submodules) in sdists by default, matching setuptools-scm behavior. The `MANIFEST.in` is not needed. Verify with `python -m build --sdist` and compare contents.

### Risk: Editable installs behave differently

**Mitigation:** meson-python supports editable installs via `pip install -e . --no-build-isolation`. The experience differs from setuptools (meson-python uses import hooks rather than `.egg-link`). The pixi-based dev workflow should still work. Test editable installs explicitly in Phase 1.

### Risk: Windows MSVC compatibility

**Mitigation:** Meson has excellent MSVC support. The SIMD flag logic (`-msse2`, `-mavx2`) is GCC/Clang-specific; MSVC equivalents (`/arch:SSE2`, `/arch:AVX2`) are needed. Meson's `cc.get_id()` can branch on compiler. Current `setup.py` already handles this partially (the `__SSE2__`/`__AVX2__` defines for `os.name == 'nt'`).

## References

- [meson-python documentation](https://mesonbuild.com/meson-python/)
- [NumPy meson migration](https://numpy.org/doc/stable/reference/distutils_status_migration.html)
- [SciPy meson migration PR](https://github.com/scipy/scipy/pull/15959)
- [Meson Cython support](https://mesonbuild.com/Cython.html)
- [Meson SIMD module](https://mesonbuild.com/Simd-module.html)
- [PR #569 — system library linking](https://github.com/zarr-developers/numcodecs/pull/569)
- [Issue #464 — use system libraries](https://github.com/zarr-developers/numcodecs/issues/464)
