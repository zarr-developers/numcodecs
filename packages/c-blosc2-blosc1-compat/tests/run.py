#!/usr/bin/env python3
"""Check that chunks written by the patched c-blosc2 can be read by C-Blosc 1.x.

Compiles the programs in tests/c against both libraries and runs, over every
compressor in {blosclz, lz4, lz4hc, zlib, zstd} x shuffle {none, byte, bit} x typesize
x buffer size (including sizes that are not a multiple of the typesize):

  global, ctx   c-blosc2 writes Blosc1 chunks through blosc1_compress / blosc2_compress_ctx,
                C-Blosc 1.x decodes them.
  b1            C-Blosc 1.x writes, c-blosc2 decodes (backward compatibility).

Example:
  run.py --blosc1-include c-blosc/blosc --blosc1-lib build-b1/blosc \\
         --blosc2-include c-blosc2/include --blosc2-ldflags "libblosc2.a libzstd.a liblz4.a -lz"
"""
import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

C = Path(__file__).resolve().parent / "c"


def sh(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blosc1-include", required=True, help="directory with blosc.h (C-Blosc 1.x)")
    ap.add_argument("--blosc1-lib", required=True, help="directory with the C-Blosc 1.x shared library")
    ap.add_argument("--blosc2-include", required=True, help="directory with blosc2.h (patched c-blosc2)")
    ap.add_argument("--blosc2-ldflags", required=True, help="libraries to link c-blosc2 and its codecs")
    ap.add_argument("--cc", default=os.environ.get("CC", "cc"))
    a = ap.parse_args()

    lib1 = Path(a.blosc1_lib).resolve()
    ld1 = ["-L" + str(lib1), "-lblosc", "-Wl,-rpath," + str(lib1)]
    ld2 = shlex.split(a.blosc2_ldflags) + ["-lpthread", "-lm"]
    inc = lambda d: "-I" + str(Path(d).resolve())  # noqa: E731

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "out").mkdir()
        cc = [a.cc, "-O1", "-I" + str(C)]
        sh(cc + [inc(a.blosc2_include), str(C / "prod.c"), "-o", "prod"] + ld2, cwd=tmp)
        sh(cc + [inc(a.blosc1_include), str(C / "cons.c"), "-o", "cons"] + ld1, cwd=tmp)
        sh(cc + [inc(a.blosc1_include), str(C / "prod1.c"), "-o", "prod1"] + ld1, cwd=tmp)
        sh(cc + [inc(a.blosc2_include), str(C / "cons2.c"), "-o", "cons2"] + ld2, cwd=tmp)
        failed = False
        for mode in ("global", "ctx"):
            sh(["./prod", mode], cwd=tmp, stdout=subprocess.DEVNULL)
            failed |= subprocess.run(["./cons", mode], cwd=tmp).returncode != 0
        sh(["./prod1"], cwd=tmp)
        failed |= subprocess.run(["./cons2", "b1"], cwd=tmp).returncode != 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
