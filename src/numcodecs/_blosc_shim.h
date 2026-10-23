/* Common C API for the two Blosc backends that blosc.pyx can be built against.
 *
 * - c-blosc 1.x (default): thin aliases for the blosc_* API.
 * - c-blosc2 (-DNUMCODECS_BLOSC2): the blosc1_*, blosc2_* API. Chunks are always written
 *   in the Blosc1 format (format version 2, 16-byte header) so that they stay readable
 *   by C-Blosc 1.x; this needs the blosc1-format patch applied to c-blosc2 (see
 *   subprojects/packagefiles/c-blosc2/blosc1-format.patch). c-blosc2 reads both formats.
 */
#ifndef NUMCODECS_BLOSC_SHIM_H
#define NUMCODECS_BLOSC_SHIM_H

#include <stdlib.h>
#include <string.h>

#ifdef NUMCODECS_BLOSC2

#include <blosc2.h>

#define NC_BLOSC_LIBRARY "c-blosc2"
#define NC_BLOSC_VERSION_STRING BLOSC2_VERSION_STRING
#define NC_BLOSC_VERSION_DATE BLOSC2_VERSION_DATE
/* Blosc1 chunk header size; what is written. Buffers are sized with NC_BLOSC_DEST_OVERHEAD. */
#define NC_BLOSC_MAX_OVERHEAD BLOSC_MIN_HEADER_LENGTH
#define NC_BLOSC_DEST_OVERHEAD BLOSC2_MAX_OVERHEAD
#define NC_BLOSC_MAX_BUFFERSIZE BLOSC2_MAX_BUFFERSIZE
#define NC_BLOSC_MAX_THREADS 256
#define NC_BLOSC_MAX_TYPESIZE BLOSC_MAX_TYPESIZE
#define NC_BLOSC_MIN_HEADER_LENGTH BLOSC_MIN_HEADER_LENGTH

#define nc_blosc_init blosc2_init
#define nc_blosc_destroy blosc2_destroy
#define nc_blosc_get_nthreads blosc2_get_nthreads
#define nc_blosc_set_nthreads blosc2_set_nthreads
#define nc_blosc_set_compressor blosc1_set_compressor
#define nc_blosc_set_blocksize blosc1_set_blocksize
#define nc_blosc_list_compressors blosc2_list_compressors
#define nc_blosc_compress blosc1_compress
#define nc_blosc_decompress blosc1_decompress
#define nc_blosc_cbuffer_sizes blosc1_cbuffer_sizes
#define nc_blosc_cbuffer_validate blosc1_cbuffer_validate
#define nc_blosc_cbuffer_complib blosc2_cbuffer_complib
#define nc_blosc_cbuffer_metainfo blosc1_cbuffer_metainfo

/* The global context writes Blosc1 chunks when this environment variable is set. It is
 * read on each compression call, so set it before the first one. */
static inline void nc_blosc_setup(void) {
#ifdef _WIN32
  _putenv_s("BLOSC_BLOSC1_COMPAT", "1");
#else
  setenv("BLOSC_BLOSC1_COMPAT", "1", 1);
#endif
}

/* Equivalent of blosc_compress_ctx: no global state, so no locking is needed. */
static inline int nc_blosc_compress_ctx(
    int clevel, int doshuffle, size_t typesize, size_t nbytes, const void* src,
    void* dest, size_t destsize, const char* compressor, size_t blocksize,
    int numinternalthreads) {
  int compcode = blosc2_compname_to_compcode(compressor);
  if (compcode < 0) {
    return -1;
  }
  blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
  cparams.compcode = (uint8_t)compcode;
  cparams.clevel = (uint8_t)clevel;
  cparams.typesize = (int32_t)typesize;
  cparams.blocksize = (int32_t)blocksize;
  cparams.nthreads = (int16_t)numinternalthreads;
  memset(cparams.filters, 0, sizeof(cparams.filters));
  memset(cparams.filters_meta, 0, sizeof(cparams.filters_meta));
  if (doshuffle == BLOSC_SHUFFLE) {
    cparams.filters[BLOSC2_MAX_FILTERS - 1] = BLOSC_SHUFFLE;
  }
  else if (doshuffle == BLOSC_BITSHUFFLE) {
    cparams.filters[BLOSC2_MAX_FILTERS - 1] = BLOSC_BITSHUFFLE;
  }
  blosc2_context* ctx = blosc2_create_cctx(cparams);
  if (ctx == NULL) {
    return -1;
  }
  int rc = blosc2_ctx_set_blosc1_format(ctx, true);
  if (rc < 0) {
    blosc2_free_ctx(ctx);
    return rc;
  }
  rc = blosc2_compress_ctx(ctx, src, (int32_t)nbytes, dest, (int32_t)destsize);
  blosc2_free_ctx(ctx);
  return rc;
}

/* Equivalent of blosc_decompress_ctx; `srcsize` is the number of valid source bytes. */
static inline int nc_blosc_decompress_ctx(
    const void* src, size_t srcsize, void* dest, size_t destsize,
    int numinternalthreads) {
  blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
  dparams.nthreads = (int16_t)numinternalthreads;
  blosc2_context* ctx = blosc2_create_dctx(dparams);
  if (ctx == NULL) {
    return -1;
  }
  int rc = blosc2_decompress_ctx(ctx, src, (int32_t)srcsize, dest, (int32_t)destsize);
  blosc2_free_ctx(ctx);
  return rc;
}

#else /* c-blosc 1.x */

#include <blosc.h>

#define NC_BLOSC_LIBRARY "c-blosc"
#define NC_BLOSC_VERSION_STRING BLOSC_VERSION_STRING
#define NC_BLOSC_VERSION_DATE BLOSC_VERSION_DATE
#define NC_BLOSC_MAX_OVERHEAD BLOSC_MAX_OVERHEAD
#define NC_BLOSC_DEST_OVERHEAD BLOSC_MAX_OVERHEAD
#define NC_BLOSC_MAX_BUFFERSIZE BLOSC_MAX_BUFFERSIZE
#define NC_BLOSC_MAX_THREADS BLOSC_MAX_THREADS
#define NC_BLOSC_MAX_TYPESIZE BLOSC_MAX_TYPESIZE
#define NC_BLOSC_MIN_HEADER_LENGTH BLOSC_MIN_HEADER_LENGTH

#define nc_blosc_init blosc_init
#define nc_blosc_destroy blosc_destroy
#define nc_blosc_get_nthreads blosc_get_nthreads
#define nc_blosc_set_nthreads blosc_set_nthreads
#define nc_blosc_set_compressor blosc_set_compressor
#define nc_blosc_set_blocksize blosc_set_blocksize
#define nc_blosc_list_compressors blosc_list_compressors
#define nc_blosc_compress blosc_compress
#define nc_blosc_decompress blosc_decompress
#define nc_blosc_cbuffer_sizes blosc_cbuffer_sizes
#define nc_blosc_cbuffer_validate blosc_cbuffer_validate
#define nc_blosc_cbuffer_complib blosc_cbuffer_complib
#define nc_blosc_cbuffer_metainfo blosc_cbuffer_metainfo
#define nc_blosc_compress_ctx blosc_compress_ctx

static inline void nc_blosc_setup(void) {}

static inline int nc_blosc_decompress_ctx(
    const void* src, size_t srcsize, void* dest, size_t destsize,
    int numinternalthreads) {
  (void)srcsize;
  return blosc_decompress_ctx(src, dest, destsize, numinternalthreads);
}

#endif

#endif /* NUMCODECS_BLOSC_SHIM_H */
