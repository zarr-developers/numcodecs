#!/usr/bin/env python3
"""Copy this package's build overlay into subprojects/packagefiles/c-blosc2.

Meson wraps only look for `patch_directory` in subprojects/packagefiles,
so that directory holds copies of what lives here. Run without arguments to update the
copies, or with --check (used in CI) to fail if they are out of date.
"""

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
DEST = PKG.parents[1] / "subprojects" / "packagefiles" / "c-blosc2"


def sources():
    for f in sorted((PKG / "meson").iterdir()):
        yield f, DEST / f.name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    check = ap.parse_args().check
    expected = {dst for _, dst in sources()}
    stale = [d for d in DEST.rglob("*") if d.is_file() and d not in expected]
    bad = [
        dst
        for src, dst in sources()
        if not dst.exists() or not filecmp.cmp(src, dst, shallow=False)
    ]
    if check:
        for f in bad + stale:
            print(f"out of sync: {f}", file=sys.stderr)
        return 1 if bad or stale else 0
    for src, dst in sources():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    for f in stale:
        f.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
