# TODO

## Check upstream c-blosc2 status of patch 0001

Patch `0001-reader-bitunshuffle-v2-leftovers.patch` is submitted upstream:

- issue: https://github.com/Blosc/c-blosc2/issues/812
- pull request: https://github.com/Blosc/c-blosc2/pull/813

Check periodically whether #813 has been merged and which c-blosc2 release contains it.
When `UPSTREAM` pins a release that includes it:

1. delete `patches/0001-reader-bitunshuffle-v2-leftovers.patch`;
2. run `python tools/sync_wrap.py` and update `diff_files` in `subprojects/c-blosc2.wrap`
   (`python tools/make_wrap_diff_files.py` prints the line);
3. update the upstream table in `README.md`.

If upstream asks for changes or rejects it, update the patch and this note to match.

Patches `0002`-`0004` (the opt-in Blosc1 writer) have not been submitted upstream.
