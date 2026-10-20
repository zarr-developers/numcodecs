"""Build the trimmed c-blosc source tarball used by ``subprojects/c-blosc.wrap``.

The upstream c-blosc archive bundles its own copies of zstd, lz4, zlib and snappy
(``internal-complibs/``), plus tests, benchmarks and compatibility fixtures. numcodecs
takes those codec libraries from their own subprojects, so shipping them again only
inflates the sdist. This script keeps just the files the meson build needs.

The output is reproducible: members are sorted and all metadata (mtime, owner,
permissions) is normalized, so the uncompressed tar is byte-identical for a given
commit. The gzip layer is also deterministic for a given zlib, but may differ across
zlib implementations, so verify a published tarball by comparing ``tar sha256``.

Usage::

    python tools/make_c_blosc_tarball.py <commit-sha> [-o OUTDIR]

Then upload the tarball as a release asset and update ``source_url``,
``source_filename``, ``source_hash`` and ``directory`` in ``subprojects/c-blosc.wrap``.
"""

import argparse
import gzip
import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path

ARCHIVE_URL = "https://github.com/Blosc/c-blosc/archive/{sha}.tar.gz"

# Paths relative to the c-blosc repository root. Directories are included recursively.
KEEP = ("blosc/", "LICENSE.txt", "LICENSES/", "README.md")


def _keep(relpath: str) -> bool:
    return any(relpath == k or (k.endswith("/") and relpath.startswith(k)) for k in KEEP)


def build(sha: str, outdir: Path) -> Path:
    with urllib.request.urlopen(ARCHIVE_URL.format(sha=sha)) as resp:
        archive = resp.read()

    src_prefix = f"c-blosc-{sha}/"
    dst_prefix = f"c-blosc-{sha[:12]}-numcodecs/"
    raw = io.BytesIO()
    with (
        tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as upstream,
        tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as out,
    ):
        files = sorted(
            (m for m in upstream.getmembers() if m.isfile() and m.name.startswith(src_prefix)),
            key=lambda m: m.name,
        )
        files = [m for m in files if _keep(m.name[len(src_prefix) :])]
        if not any(m.name.endswith("/blosc/blosc.c") for m in files):
            raise RuntimeError("blosc/blosc.c not found; unexpected archive layout")
        for m in files:
            info = tarfile.TarInfo(dst_prefix + m.name[len(src_prefix) :])
            info.size = m.size
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            out.addfile(info, upstream.extractfile(m))
    tar_bytes = raw.getvalue()

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{dst_prefix.rstrip('/')}.tar.gz"
    gz = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=gz, compresslevel=9, mtime=0) as f:
        f.write(tar_bytes)
    path.write_bytes(gz.getvalue())

    print(f"wrote      {path} ({path.stat().st_size} bytes, {len(files)} files)")
    print(f"directory  {dst_prefix.rstrip('/')}")
    print(f"sha256     {hashlib.sha256(gz.getvalue()).hexdigest()}  (source_hash)")
    print(f"tar sha256 {hashlib.sha256(tar_bytes).hexdigest()}  (uncompressed, for verification)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sha", help="full c-blosc commit sha")
    parser.add_argument("-o", "--outdir", type=Path, default=Path("."))
    args = parser.parse_args()
    if len(args.sha) != 40:
        parser.error("pass the full 40-character commit sha")
    build(args.sha, args.outdir)


if __name__ == "__main__":
    main()
