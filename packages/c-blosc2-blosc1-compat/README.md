# c-blosc2-blosc1-compat

A build overlay for [c-blosc2](https://github.com/Blosc/c-blosc2) that makes it usable as the
backend of numcodecs' `Blosc` codec: a meson build (upstream only has CMake) for a c-blosc2
branch that can **write chunks that C-Blosc 1.x can decode**. Stock c-blosc2 cannot, even
through its `blosc1_*` API or with `BLOSC_BLOSC1_COMPAT` (its README says so, and
`tests/run.py` confirms it).

This directory is self-contained (it does not import anything from numcodecs) so it can be
split out with `git subtree split` if it ever needs to live in its own repository.

## Contents

| Path | What |
|---|---|
| `UPSTREAM` | the c-blosc2 branch and commit in use; `subprojects/c-blosc2.wrap` must match |
| `meson/` | `meson.build` and `meson.options` for c-blosc2 |
| `tests/` | the forward-compatibility check, see below |
| `tools/` | `sync_wrap.py` |
| `TODO.md` | what to do once the c-blosc2 work is released |

## The c-blosc2 branch

`subprojects/c-blosc2.wrap` clones the `blosc1-format-writer` branch of
[mkitti/c-blosc2](https://github.com/mkitti/c-blosc2), pinned to a commit. The branch is
upstream c-blosc2 plus the change proposed in
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
[#813](https://github.com/Blosc/c-blosc2/pull/813)) is already in upstream c-blosc2 and so
in the branch.

## Wiring into the build

Meson only looks for `patch_directory` in `subprojects/packagefiles`, so
`subprojects/packagefiles/c-blosc2/` holds a copy of `meson/`. After editing here run
`python tools/sync_wrap.py`; CI runs it with `--check`.

## Testing

`tests/run.py` builds small C programs against C-Blosc 1.x and c-blosc2 and checks 735
cases in three directions (see its docstring for the command line): c-blosc2 global-context
to C-Blosc 1.x, c-blosc2 context API to C-Blosc 1.x, and C-Blosc 1.x to c-blosc2. All must
decode. The c-blosc2 branch has its own unit tests (`tests/test_blosc1_format.c`).

## Updating c-blosc2

Edit `UPSTREAM` and the `revision` (and `url`, if the branch moves) in
`subprojects/c-blosc2.wrap` together; a test checks that they agree. Then run
`python tools/sync_wrap.py` and the tests. If upstream changes its source file list, update
`meson/meson.build`.

## Upstreaming

The wrap depends on a personal fork until Blosc/c-blosc2#815 is merged and released; see
`TODO.md`.
