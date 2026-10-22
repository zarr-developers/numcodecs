# TODO

## Move the wrap off the fork once Blosc/c-blosc2#815 is released

`subprojects/c-blosc2.wrap` clones the `blosc1-format-writer` branch of
https://github.com/mkitti/c-blosc2 (see `UPSTREAM`). The change is proposed upstream in
https://github.com/Blosc/c-blosc2/pull/815 (it builds on #812 / #813, already merged).

Check periodically whether #815 has been merged and which c-blosc2 release contains it.
When there is such a release:

1. point the wrap at it: `url = https://github.com/Blosc/c-blosc2.git`, `revision = <tag or
   commit>` (or switch back to a `[wrap-file]` with `source_url` and `source_hash`);
2. update `UPSTREAM` to match, and check that `meson/meson.build` still lists the right
   source files;
3. run `python tools/sync_wrap.py` and `tests/run.py`;
4. update `docs/release.md` and `README.md`.

If upstream asks for changes (for example to the `BLOSC_BLOSC1_COMPAT` behavior) or rejects
it, update the branch, `UPSTREAM` and the wrap's `revision` to match.

Do not release numcodecs with the wrap pointing at the fork: a personal branch is not a
stable build dependency.
