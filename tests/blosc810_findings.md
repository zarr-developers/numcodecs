# numcodecs#810 investigation findings

Env: numcodecs 0.15.1, blosc clib 1.21.6, zarr 2.18.2, xarray 2026.4.0, dask 2026.3.0.
Source codec: `Blosc(cname='lz4', clevel=5, shuffle=SHUFFLE, blocksize=0)`, float32, chunk (1080,1440).

## Error
`RuntimeError: error during blosc decompression: -1` (numcodecs/blosc.pyx:414),
intermittent, during distributed (coiled) reads of a blosc-compressed zarr on S3.
Reported mitigations BLOSC_NTHREADS=1 / BLOSC_NOLOCK=1 / NUMEXPR_NUM_THREADS=1 did not help.

## What did NOT reproduce it
Standalone `Blosc.decode()` hammered concurrently (no S3, no dask, no zarr):
- 48 threads x 3000 iters x 8 chunks = 144k concurrent decodes, blosc nthreads=1: clean.
- 32 threads x 2000 iters, blosc internal nthreads=4 and =8: clean.
=> Concurrent in-memory decode of intact buffers is thread-safe here. The race
theory (numcodecs/blosc thread-safety) is not supported by these runs.

## What DID reproduce the exact error (deterministic)
Feeding blosc a **truncated** compressed buffer reproduces the identical error:
    enc = codec.encode(float32 (1080,1440))         # len 4,944,670 bytes
    codec.decode(enc[:int(len*0.99)])  -> RuntimeError: error during blosc decompression: -1
    codec.decode(enc[:int(len*0.50)])  -> same
    codec.decode(enc[:int(len*0.999)]) -> same

## Hypothesis
The `-1` is blosc correctly rejecting an **incomplete/truncated input buffer**. The bug
is therefore upstream of the codec: under high read concurrency, the chunk bytes handed
to blosc are occasionally short (a partial / mis-retried S3 range request, or a
connection-pool race in s3fs/fsspec/zarr). This explains:
  - BLOSC_NTHREADS=1 not helping (not a blosc thread bug),
  - failures clustering near completion of large jobs (more chunk reads = more chances),
  - the absence of any pure-codec reproducer.