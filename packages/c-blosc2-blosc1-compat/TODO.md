# TODO

## Move the wrap to a c-blosc2 release

`subprojects/c-blosc2.wrap` clones https://github.com/Blosc/c-blosc2 at a commit on `main`
(see `UPSTREAM`) that includes the Blosc1 format writer, merged in
https://github.com/Blosc/c-blosc2/pull/815 (it builds on #812 / #813).

Check which c-blosc2 release first contains #815. When there is one:

1. point the wrap's `revision` at its tag (or switch to a `[wrap-file]` with `source_url` and
   `source_hash`);
2. update `UPSTREAM` to match, and check that `meson/meson.build` still lists the right
   source files;
3. run `python tools/sync_wrap.py` and `tests/run.py`;
4. update `docs/release.md` and `README.md`.

Do not release numcodecs with the wrap pointing at an unreleased commit of `main`.
