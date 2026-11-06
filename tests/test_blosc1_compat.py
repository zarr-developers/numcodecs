"""Compatibility of numcodecs' Blosc with C-Blosc 1.x, using python-blosc as the reference.

python-blosc (the ``blosc`` package) bundles C-Blosc 1.x, so it stands in for every other
reader and writer of the Blosc1 format, among them the Blosc codec of zarr v3. Whatever
backend ``numcodecs.blosc`` is built with, these must hold:

* chunks written by numcodecs decode with C-Blosc 1.x to the original bytes (forward);
* chunks written by C-Blosc 1.x decode with numcodecs to the original bytes (backward);
* numcodecs round-trips its own chunks, and its chunks have a sound Blosc1 header.

The compressed bytes are *not* expected to be identical to what C-Blosc 1.x writes (the
libraries choose different default block sizes, for one); the decoded bytes are.

This module is optional. It is skipped unless ``blosc`` and ``hypothesis`` are installed,
and it is only run by .github/workflows/blosc1-compat.yaml. To leave it out anywhere else
run ``pytest -m "not blosc1_compat"``.  The number of generated examples can be set with
the BLOSC1_COMPAT_EXAMPLES environment variable.
"""

import contextlib
import itertools
import os

import numpy as np
import pytest

pytest.importorskip("hypothesis")
blosc1 = pytest.importorskip("blosc", reason="python-blosc (C-Blosc 1.x) is the reference")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from numcodecs import blosc as nc_blosc  # noqa: E402

pytestmark = pytest.mark.blosc1_compat

EXAMPLES = int(os.environ.get("BLOSC1_COMPAT_EXAMPLES", "300"))
MAX_TYPESIZE = 255  # C-Blosc stores a larger typesize as 1

# compressors that both libraries have: a build can leave some out (e.g. zstd, zlib)
CNAMES = sorted(
    set(nc_blosc.list_compressors())
    & set(blosc1.compressor_list())
    & {"blosclz", "lz4", "lz4hc", "zlib", "zstd"}
)
SHUFFLES = [nc_blosc.NOSHUFFLE, nc_blosc.SHUFFLE, nc_blosc.BITSHUFFLE]

# the sizes where things change: the buffer minimum (32), element counts around multiples
# of 8 (bitshuffle) and around 128 (the C-Blosc 1.x split rule), and typesizes around 16
# (the largest splittable one) and 255
EDGE_ITEMSIZES = [1, 2, 3, 4, 5, 7, 8, 12, 16, 17, 24, 32, 255, 256, 300]
EDGE_ELEMENTS = [0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1023, 1024]
EDGE_BLOCKSIZES = [0, 1, 7, 32, 33, 100, 128, 1000, 4096, 2**15 + 1, 2**16, 2**18, 2**20]
KINDS = ["zeros", "constant", "random", "ramp", "periodic", "sparse", "sorted"]

SETTINGS = settings(
    max_examples=EXAMPLES,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    print_blob=True,
)


def make_payload(kind, seed, itemsize, nelements, leftover):
    """Bytes of ``nelements`` items of ``itemsize`` bytes plus ``leftover`` extra bytes."""
    rng = np.random.default_rng(seed)
    items = np.zeros((nelements, itemsize), dtype=np.uint8)
    if kind == "constant":
        items[:] = rng.integers(1, 256, dtype=np.uint8)
    elif kind == "random":
        items[:] = rng.integers(0, 256, size=items.shape, dtype=np.uint8)
    elif kind in ("ramp", "sorted"):
        # integers growing by a small step, little-endian across the item's bytes
        step = rng.integers(0, 4 if kind == "ramp" else 2**20, size=nelements, dtype=np.uint64)
        values = np.cumsum(step, dtype=np.uint64)
        for j in range(min(itemsize, 8)):
            items[:, j] = (values >> np.uint64(8 * j)) & np.uint64(0xFF)
    elif kind == "periodic":
        period = rng.integers(1, 9)
        pattern = rng.integers(0, 256, size=(period, itemsize), dtype=np.uint8)
        items[:] = pattern[np.arange(nelements) % period]
    elif kind == "sparse":
        mask = rng.random(items.shape) < 0.02
        items[mask] = rng.integers(0, 256, size=int(mask.sum()), dtype=np.uint8)
    tail = rng.integers(0, 256, size=leftover, dtype=np.uint8)
    return np.concatenate([items.reshape(-1), tail]).tobytes()


@st.composite
def payloads(draw):
    itemsize = draw(st.sampled_from(EDGE_ITEMSIZES) | st.integers(1, 300))
    nelements = draw(st.sampled_from(EDGE_ELEMENTS) | st.integers(0, 4096))
    # the bytes that do not make up a whole item are copied through unfiltered
    leftover = draw(st.sampled_from([0, 0, 1, itemsize - 1, itemsize // 2]))
    leftover = max(0, min(leftover, itemsize - 1))
    kind = draw(st.sampled_from(KINDS))
    seed = draw(st.integers(0, 2**32 - 1))
    return itemsize, make_payload(kind, seed, itemsize, nelements, leftover)


compress_params = st.fixed_dictionaries(
    {
        "cname": st.sampled_from(CNAMES),
        "clevel": st.integers(0, 9),
        "shuffle": st.sampled_from(SHUFFLES),
        "blocksize": st.sampled_from(EDGE_BLOCKSIZES) | st.integers(0, 2**20),
    }
)


def v1_typesize(itemsize):
    return itemsize if itemsize <= MAX_TYPESIZE else 1


def v1_compress(data, itemsize, cname, clevel, shuffle, blocksize):
    """Compress with C-Blosc 1.x."""
    blosc1.set_nthreads(1)
    blosc1.set_blocksize(blocksize)
    try:
        return blosc1.compress(
            data, typesize=v1_typesize(itemsize), clevel=clevel, shuffle=shuffle, cname=cname
        )
    finally:
        blosc1.set_blocksize(0)


@contextlib.contextmanager
def numcodecs_path(path):
    """Make numcodecs compress and decompress through one of its two code paths.

    numcodecs uses blosc's global context when it may use threads, which is the default in
    the main thread, and a context of its own otherwise (``use_threads = False``, other
    threads, forked processes). With the c-blosc2 backend the two differ in how they ask for
    Blosc1 chunks, so both have to be checked. ``use_threads`` is a global, so this is a
    context manager and not a (function-scoped) fixture, which hypothesis does not allow.
    """
    saved = nc_blosc.use_threads
    nc_blosc.use_threads = path == "global"
    try:
        yield
    finally:
        nc_blosc.use_threads = saved


PATHS = pytest.mark.parametrize("path", ["global", "context"])


def nc_compress(data, itemsize, cname, clevel, shuffle, blocksize):
    """Compress with numcodecs, the way zarr does: bytes and an explicit typesize."""
    return nc_blosc.Blosc(
        cname=cname, clevel=clevel, shuffle=shuffle, blocksize=blocksize, typesize=itemsize
    ).encode(np.frombuffer(data, dtype=np.uint8))


def check_header(chunk, data, itemsize):
    """A chunk is a sound Blosc1 chunk of ``data``, whoever wrote it."""
    assert chunk[0] == 2, "format version is not the Blosc1 one"
    nbytes, cbytes, blocksize = blosc1.get_cbuffer_sizes(chunk)
    assert nbytes == len(data)
    assert cbytes == len(chunk)
    if data:
        assert blocksize <= len(data)
        assert chunk[3] == v1_typesize(itemsize)


@PATHS
@SETTINGS
@given(payload=payloads(), params=compress_params)
def test_decodes_in_c_blosc1(path, payload, params):
    """Forward: what numcodecs writes, C-Blosc 1.x reads."""
    itemsize, data = payload
    with numcodecs_path(path):
        chunk = nc_compress(data, itemsize, **params)
    check_header(chunk, data, itemsize)
    assert blosc1.decompress(chunk) == data


@PATHS
@SETTINGS
@given(payload=payloads(), params=compress_params)
def test_decodes_from_c_blosc1(path, payload, params):
    """Backward: what C-Blosc 1.x writes, numcodecs reads."""
    itemsize, data = payload
    chunk = v1_compress(data, itemsize, **params)
    with numcodecs_path(path):
        assert bytes(nc_blosc.decompress(chunk)) == data


@PATHS
@SETTINGS
@given(payload=payloads(), params=compress_params)
def test_numcodecs_roundtrip(path, payload, params):
    itemsize, data = payload
    with numcodecs_path(path):
        chunk = nc_compress(data, itemsize, **params)
        assert bytes(nc_blosc.decompress(chunk)) == data
        # the codec without a typesize takes it from the array, here bytes
        codec = nc_blosc.Blosc(**params)
        assert bytes(codec.decode(codec.encode(np.frombuffer(data, dtype=np.uint8)))) == data


@PATHS
@pytest.mark.parametrize(
    ("itemsize", "dtype"), [(1, "u1"), (2, "u2"), (4, "f4"), (8, "f8"), (16, "c16")]
)
@SETTINGS
@given(
    params=compress_params,
    nelements=st.sampled_from(EDGE_ELEMENTS) | st.integers(0, 2048),
    seed=st.integers(0, 2**32 - 1),
)
def test_typed_arrays(path, itemsize, dtype, params, nelements, seed):
    """Arrays give numcodecs their typesize themselves, without a typesize argument."""
    data = make_payload("random" if seed % 2 else "ramp", seed, itemsize, nelements, 0)
    array = np.frombuffer(data, dtype=dtype)
    with numcodecs_path(path):
        chunk = nc_blosc.compress(
            array,
            params["cname"].encode(),
            params["clevel"],
            params["shuffle"],
            # a blocksize is not an array property, so ask for the automatic one here
            0,
        )
        assert bytes(nc_blosc.decompress(chunk)) == data
    check_header(chunk, data, itemsize)
    assert blosc1.decompress(chunk) == data


# A deterministic grid for the same, so that a failure names its parameters and runs the
# same everywhere: every compressor x shuffle x typesize x size, in both directions
GRID = list(
    itertools.product(
        CNAMES,
        SHUFFLES,
        [1, 2, 3, 4, 7, 8, 16, 17, 255, 300],
        [0, 1, 31, 32, 33, 127, 1001, 4096, 65537, 100003],
    )
)


@PATHS
@pytest.mark.parametrize(("cname", "shuffle", "itemsize", "nbytes"), GRID)
def test_grid(path, cname, shuffle, itemsize, nbytes):
    data = make_payload("ramp", 1234, itemsize, nbytes // itemsize, nbytes % itemsize)
    assert len(data) == nbytes
    params = {"cname": cname, "clevel": 5, "shuffle": shuffle, "blocksize": 0}

    with numcodecs_path(path):
        chunk = nc_compress(data, itemsize, **params)
    check_header(chunk, data, itemsize)
    assert blosc1.decompress(chunk) == data

    chunk = v1_compress(data, itemsize, **params)
    with numcodecs_path(path):
        assert bytes(nc_blosc.decompress(chunk)) == data
