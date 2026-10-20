#!/usr/bin/env python3
"""Check that the patch series applies to a c-blosc2 commit or tag.

Usage: rebase.py [REF]   (default: the commit in UPSTREAM)
"""
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else dict(
        line.split(" = ") for line in (PKG / "UPSTREAM").read_text().split("\n") if " = " in line
    )["commit"]
    url = f"https://github.com/Blosc/c-blosc2/archive/{ref}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        tar = Path(tmp) / "src.tar.gz"
        urllib.request.urlretrieve(url, tar)
        with tarfile.open(tar) as t:
            t.extractall(tmp, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
        (src,) = [d for d in Path(tmp).iterdir() if d.is_dir()]
        status = 0
        for patch in sorted((PKG / "patches").glob("*.patch")):
            r = subprocess.run(
                ["patch", "-p1", "--no-backup-if-mismatch", "-i", str(patch)],
                cwd=src, capture_output=True, text=True,
            )
            print(f"{'ok  ' if r.returncode == 0 else 'FAIL'} {patch.name}")
            if r.returncode:
                print(r.stdout + r.stderr)
                status = 1
                break
    return status


if __name__ == "__main__":
    sys.exit(main())
