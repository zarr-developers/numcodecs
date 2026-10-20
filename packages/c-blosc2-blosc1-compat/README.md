# c-blosc2-blosc1-compat

An overlay for [c-blosc2](https://github.com/Blosc/c-blosc2) that makes it usable as the
backend of numcodecs' `Blosc` codec: a meson build, plus a small patch series that lets
c-blosc2 **write chunks that C-Blosc 1.x can decode**. Stock c-blosc2 cannot, even through
its `blosc1_*` API or with `BLOSC_BLOSC1_COMPAT` (its README says so, and
`tests/run.py` confirms it).

This directory is self-contained (it does not import anything from numcodecs) so it can be
split out with `git subtree split` if it ever needs to live in its own repository, or be
deleted if upstream accepts the patches.

## Contents

| Path | What |
|---|---|
| `UPSTREAM` | the pinned c-blosc2 commit and source hash; `subprojects/c-blosc2.wrap` must match |
| `patches/` | the patch series, applied in order; one concern per patch |
| `meson/` | `meson.build` and `meson.options` for c-blosc2 (upstream only has CMake) |
| `tests/` | the forward-compatibility check, see below |
| `tools/` | `sync_wrap.py`, `rebase.py` |

## The patches

Format version 2 (Blosc1) differs from what c-blosc2 writes in three ways that matter to a
reader, plus one reader bug:

1. `0001` reader: bytes left over after a bit-shuffled version-2 block were not copied.
   A plain bug fix, independent of the rest.
2. `0002` an opt-in flag: write the 16-byte Blosc1 header with format version 2. Set with
   `blosc2_ctx_set_blosc1_format()` or, for the global context, `BLOSC_BLOSC1_COMPAT`.
   Off by default, no ABI change.
3. `0003` bitshuffle: version 2 shuffles a block only if its element count is a multiple
   of 8, and stores it unshuffled otherwise.
4. `0004` block splitting: use the C-Blosc 1.x rule, which its reader recomputes.

Only plain shuffle/bitshuffle with blosclz, lz4, lz4hc, zlib and zstd are supported in this
mode. The output decodes with C-Blosc 1.x but is not byte-identical to what it writes.

## Wiring into the build

Meson only looks for `patch_directory` and `diff_files` in `subprojects/packagefiles`, so
`subprojects/packagefiles/c-blosc2/` holds copies of `meson/` and `patches/`. After editing
here run `python tools/sync_wrap.py`; CI runs it with `--check`. If a patch is added or
removed, update `diff_files` in `subprojects/c-blosc2.wrap`
(`tools/make_wrap_diff_files.py` prints the line).

## Testing

`tests/run.py` builds small C programs against C-Blosc 1.x and the patched c-blosc2 and
checks 735 cases in three directions (see its docstring for the command line):
c-blosc2 global-context to C-Blosc 1.x, c-blosc2 context API to C-Blosc 1.x, and C-Blosc 1.x
to c-blosc2. All must decode.

## Updating c-blosc2

`python tools/rebase.py <commit-or-tag>` downloads that c-blosc2, tries to apply the
series, and reports which patch stops applying. To bump: edit `UPSTREAM` and
`subprojects/c-blosc2.wrap` (directory, URL, hash), fix any patch that fails, re-export it
as a unified diff against the pristine tree (`diff -ruN a b`), run `tools/sync_wrap.py`
and the tests. The scheduled workflow `blosc2-compat-rebase` does the first step weekly.

## Upstreaming

Track each patch here when a PR exists, and delete the patch once it is in a release
that `UPSTREAM` pins:

| Patch | Upstream |
|---|---|
| 0001 | not yet submitted |
| 0002–0004 | not yet submitted |
