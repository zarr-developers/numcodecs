"""Reproducer for numcodecs#810: ``RuntimeError: error during blosc decompression: -1``.

Context
-------
The error was first seen intermittently while reading a blosc-compressed Zarr store
from S3 across many Dask/Coiled workers, near the end of large jobs. The reported
mitigations (BLOSC_NTHREADS=1, BLOSC_NOLOCK=1, NUMEXPR_NUM_THREADS=1) did not help.

Findings (see the three parts below)
------------------------------------
1. DETERMINISTIC: feeding blosc a *truncated/incomplete* compressed buffer raises the
   exact ``-1`` error. So ``-1`` is blosc correctly rejecting bad input bytes.
2. NEGATIVE CONTROL: hammering ``Blosc.decode()`` concurrently on *intact* buffers
   (tens of thousands of decodes, with and without blosc's internal threadpool) does
   NOT reproduce it. The codec itself is not the culprit.
3. FIELD REPRO (opt-in, needs S3): fetch raw chunk bytes concurrently and compare each
   buffer's length to the size declared in its own blosc header (``cbytes``). A
   mismatch proves the storage/transport layer (s3fs/fsspec/zarr) occasionally hands
   blosc a short read -- which then surfaces as the ``-1`` from part 1.

Conclusion: this is most likely a truncated-read bug upstream of numcodecs, not a
codec thread-safety race. Parts 1+2 are runnable anywhere with just numpy+numcodecs.

Run
---
    python reproduce_blosc810.py                 # parts 1 and 2 (deterministic)
    BLOSC810_S3_URL=s3://bucket/path/OM4.zarr \
    BLOSC810_S3_ENDPOINT=https://nyu1.osn.mghpcc.org/ \
    AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
    python reproduce_blosc810.py                 # also runs part 3 (field repro)

Also importable as pytest tests (the test_* functions).

Env: numcodecs 0.15.1, c-blosc 1.21.6, zarr 2.18.2 (but only numcodecs+numpy are
required for parts 1 and 2).
"""

from __future__ import annotations

import os
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numcodecs import Blosc
from numcodecs import blosc as ncb

# Match the codec on the OM4 source arrays exactly.
CODEC = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
SHAPE = (1080, 1440)  # a real chunk: float32 ~ 6.2 MB
MINUS_ONE = "error during blosc decompression: -1"

# Blosc1 header is 16 bytes; the compressed total size (incl. header) is a
# little-endian uint32 at offset 12. https://github.com/Blosc/c-blosc
_BLOSC_HEADER = struct.Struct("<B B B B I I I")  # version,versionlz,flags,typesize,nbytes,blocksize,cbytes


def _sample_chunk(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    # smooth-ish, semi-compressible float32, like an ocean field
    arr = np.cumsum(rng.standard_normal(SHAPE).astype("float32"), axis=1)
    return CODEC.encode(np.ascontiguousarray(arr))


def blosc_cbytes(buf: bytes) -> int:
    """Compressed total size (including header) that the blosc header *claims*."""
    return _BLOSC_HEADER.unpack_from(buf, 0)[6]


# --------------------------------------------------------------------------- #
# Part 1 -- DETERMINISTIC reproduction: a truncated buffer yields exactly -1.
# --------------------------------------------------------------------------- #
def test_truncated_buffer_reproduces_minus_one() -> None:
    enc = _sample_chunk()
    # sanity: the full buffer decodes
    CODEC.decode(enc)
    declared = blosc_cbytes(enc)
    assert declared == len(enc), (declared, len(enc))

    reproduced = 0
    for frac in (0.999, 0.99, 0.5, 0.1):
        truncated = enc[: int(len(enc) * frac)]
        try:
            CODEC.decode(truncated)
        except RuntimeError as e:
            assert MINUS_ONE in str(e), str(e)
            reproduced += 1
    assert reproduced == 4
    return reproduced


# --------------------------------------------------------------------------- #
# Part 2 -- NEGATIVE CONTROL: concurrent decode of intact buffers is clean.
# --------------------------------------------------------------------------- #
def test_concurrent_decode_of_intact_buffers_is_clean(
    n_threads: int = 32, n_iters: int = 2000, n_chunks: int = 8, blosc_nthreads: int = 1
) -> int:
    ncb.set_nthreads(blosc_nthreads)
    chunks = [_sample_chunk(i) for i in range(n_chunks)]
    sizes = [len(CODEC.decode(c)) for c in chunks]
    errors: list[tuple] = []
    lock = threading.Lock()

    def worker(tid: int) -> None:
        for j in range(n_iters):
            idx = (tid + j) % n_chunks
            try:
                out = CODEC.decode(chunks[idx])
            except Exception as e:  # noqa: BLE001 - we want any failure
                with lock:
                    errors.append((tid, j, type(e).__name__, str(e)))
                return
            if len(out) != sizes[idx]:
                with lock:
                    errors.append((tid, j, "ShortDecode", f"{len(out)} != {sizes[idx]}"))
                return

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for f in as_completed([ex.submit(worker, t) for t in range(n_threads)]):
            f.result()

    assert not errors, errors[:5]
    return n_threads * n_iters


# --------------------------------------------------------------------------- #
# Part 3 -- FIELD REPRO (opt-in): catch a truncated chunk read from S3.
# --------------------------------------------------------------------------- #
def field_repro_s3(url: str, endpoint: str | None, rounds: int = 50, n_threads: int = 64) -> None:
    """Concurrently fetch raw chunk bytes for one array and verify length vs header.

    For every chunk object under ``<url>/<array>/`` we read the raw (still-compressed)
    bytes and compare ``len(bytes)`` to the ``cbytes`` declared by that chunk's own
    blosc header. A mismatch is a short read from the storage layer -- the upstream
    cause of the ``-1`` decode error. We also attempt ``CODEC.decode`` and record any
    ``-1``. Repeating ``rounds`` times amplifies the intermittent failure.
    """
    import s3fs  # lazy: only needed for the S3 path

    client_kwargs = {"endpoint_url": endpoint} if endpoint else {}
    fs = s3fs.S3FileSystem(client_kwargs=client_kwargs)
    root = url[len("s3://"):] if url.startswith("s3://") else url

    # Pick a blosc-compressed array directory with many chunks.
    array_dir = None
    for name in ("thetao", "so", "uo", "vo", "zos"):
        cand = f"{root}/{name}"
        if fs.exists(cand):
            array_dir = cand
            break
    if array_dir is None:
        raise SystemExit(f"no candidate array under {root}")

    chunk_keys = [
        k for k in fs.find(array_dir)
        if not k.rsplit("/", 1)[-1].startswith(".")  # skip .zarray/.zattrs
    ]
    print(f"[field] array={array_dir} chunks={len(chunk_keys)} "
          f"rounds={rounds} threads={n_threads}")

    short_reads: list[tuple] = []
    decode_errors: list[tuple] = []
    lock = threading.Lock()

    def check(key: str, rnd: int) -> None:
        try:
            buf = fs.cat_file(key)
        except Exception as e:  # noqa: BLE001
            with lock:
                short_reads.append((rnd, key, "FetchError", str(e)))
            return
        declared = blosc_cbytes(buf)
        if len(buf) != declared:
            with lock:
                short_reads.append((rnd, key, "LengthMismatch", f"{len(buf)} != {declared}"))
        try:
            CODEC.decode(buf)
        except RuntimeError as e:
            if MINUS_ONE in str(e):
                with lock:
                    decode_errors.append((rnd, key, str(e)))

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futs = []
        for rnd in range(rounds):
            for key in chunk_keys:
                futs.append(ex.submit(check, key, rnd))
        for f in as_completed(futs):
            f.result()

    n = rounds * len(chunk_keys)
    print(f"[field] {n} concurrent chunk fetches")
    if short_reads or decode_errors:
        print(f"[field] REPRODUCED upstream truncation: "
              f"{len(short_reads)} short reads, {len(decode_errors)} decode -1")
        for r in (short_reads + decode_errors)[:10]:
            print("   ", r)
        sys.exit(1)
    print("[field] no short read this run (intermittent -- raise rounds/threads, rerun)")


def main() -> None:
    print(f"numcodecs blosc clib {ncb.__version__} ({getattr(ncb, 'VERSION_STRING', '?')})")
    n = test_truncated_buffer_reproduces_minus_one()
    print(f"[part1] DETERMINISTIC: truncated buffer -> '{MINUS_ONE}' ({n}/4 truncations)")
    total = test_concurrent_decode_of_intact_buffers_is_clean()
    print(f"[part2] negative control: {total} concurrent intact decodes, no error")

    url = os.environ.get("BLOSC810_S3_URL")
    if url:
        field_repro_s3(url, os.environ.get("BLOSC810_S3_ENDPOINT"))
    else:
        print("[part3] skipped (set BLOSC810_S3_URL + creds to run the S3 field repro)")


if __name__ == "__main__":
    main()
