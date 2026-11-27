# c-blosc2-blosc1-compat

A build overlay for [c-blosc2](https://github.com/Blosc/c-blosc2) that makes it usable as the
backend of numcodecs' `Blosc` codec: a meson build (upstream only has CMake) for a c-blosc2
that can **write chunks that C-Blosc 1.x can decode**. Stock c-blosc2 cannot, even
through its `blosc1_*` API or with `BLOSC_BLOSC1_COMPAT` (its README says so, and
`tests/run.py` confirms it).

This directory is self-contained (it does not import anything from numcodecs) so it can be
split out with `git subtree split` if it ever needs to live in its own repository.

## Contents

| Path | What |
|---|---|
| `UPSTREAM` | the c-blosc2 commit in use; `subprojects/c-blosc2.wrap` must match |
| `meson/` | `meson.build` and `meson.options` for c-blosc2 |
| `tests/` | the forward-compatibility check, see below |
| `tools/` | `sync_wrap.py` |
| `TODO.md` | moving the pin to a c-blosc2 release |

## The c-blosc2 version

`subprojects/c-blosc2.wrap` clones [Blosc/c-blosc2](https://github.com/Blosc/c-blosc2) at a
commit on `main`. It includes the change merged in
[Blosc/c-blosc2#815](https://github.com/Blosc/c-blosc2/pull/815):
`blosc2_ctx_set_blosc1_format()`, which makes a compression context write format version 2
(Blosc1) chunks. Format version 2 differs from what c-blosc2 writes in three ways that
matter to a reader:

1. the format version byte and the 16-byte header;
2. bitshuffle: version 2 shuffles a block only if its element count is a multiple of 8, and
   stores it unshuffled otherwise;
3. block splitting: a C-Blosc 1.x reader recomputes it from the typesize and blocksize.

Only plain shuffle/bitshuffle with blosclz, lz4, lz4hc, zlib and zstd are supported in this
mode. The output decodes with C-Blosc 1.x but is not byte-identical to what it writes.

The reader fix for bit-shuffled version 2 blocks with leftover bytes
([#812](https://github.com/Blosc/c-blosc2/issues/812),
[#813](https://github.com/Blosc/c-blosc2/pull/813)) is also in upstream c-blosc2.

## Wiring into the build

Meson only looks for `patch_directory` in `subprojects/packagefiles`, so
`subprojects/packagefiles/c-blosc2/` holds a copy of `meson/`. After editing here run
`python tools/sync_wrap.py`; CI runs it with `--check`.

## Testing

`tests/run.py` builds small C programs against C-Blosc 1.x and c-blosc2 and checks 735
cases in three directions (see its docstring for the command line): c-blosc2 global-context
to C-Blosc 1.x, c-blosc2 context API to C-Blosc 1.x, and C-Blosc 1.x to c-blosc2. All must
decode. c-blosc2 has its own unit tests (`tests/test_blosc1_format.c`).

## Compatibility CI

`tests/test_blosc1_compat.py` (repository `tests/`, not this directory) checks that chunks
written by numcodecs decode with C-Blosc 1.x and the reverse, with python-blosc, which
bundles C-Blosc 1.x, as the reference. It combines property-based tests (hypothesis) over
compressors, levels, shuffles, typesizes up to 300, buffer sizes around the edges that
matter (32 bytes, multiples of 8 and 128 elements, bytes left over after the last item),
block sizes and data kinds, with an exhaustive grid. Both numcodecs code paths are checked:
the one through blosc's global context and the one through a context of its own. The
compressed bytes are not expected to equal what C-Blosc 1.x writes (the libraries pick
different default block sizes); the decoded bytes are.

`.github/workflows/blosc1-compat.yaml` runs it on the c-blosc2 backend on Linux (x86_64 and
arm64), macOS and Windows, and on the default c-blosc 1.x backend as a control. It runs 300
examples per property on a pull request and 5000 weekly. It is independent of everything
else and can be turned off by

- setting the repository variable `BLOSC1_COMPAT_CI` to `false` (no code change),
- deleting the workflow file (the test module skips itself when python-blosc or hypothesis
  is not installed, so no other workflow changes), or
- `pytest -m "not blosc1_compat"` anywhere else.

To run it locally: `pip install blosc hypothesis`, then
`pytest tests/test_blosc1_compat.py` (`BLOSC1_COMPAT_EXAMPLES` sets the examples).

## Updating c-blosc2

Edit `UPSTREAM` and the `revision` in
`subprojects/c-blosc2.wrap` together; a test checks that they agree. Then run
`python tools/sync_wrap.py` and the tests. If upstream changes its source file list, update
`meson/meson.build`.

## Upstreaming

#815 is merged but not yet in a c-blosc2 release; the wrap uses a commit on `main` until it
is. See `TODO.md`.
