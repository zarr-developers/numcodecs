// producer: c-blosc2, mode: "global" (blosc1_compress, compat env), "global_noenv", "ctx"
#include <blosc2.h>
#include "common.h"
int main(int argc, char **argv) {
  const char *mode = argv[1];
  if (!strcmp(mode, "global")) setenv("BLOSC_BLOSC1_COMPAT", "TRUE", 1);
  blosc2_init(); blosc2_set_nthreads(1); if (getenv("SPLITMODE")) blosc1_set_splitmode(atoi(getenv("SPLITMODE")));
  int n = 0;
  for (int c = 0; c < 5; c++) for (int s = 0; s < 3; s++) for (int t = 0; t < 7; t++) for (int z = 0; z < 7; z++) {
    size_t nb = sizes[z]; int ts = typesizes[t];

    uint8_t *src = malloc(nb), *dst = malloc(nb + BLOSC2_MAX_OVERHEAD);
    fill(src, nb, ts, 1234);
    int cb;
    if (!strcmp(mode, "ctx")) {
      blosc2_cparams cp = BLOSC2_CPARAMS_DEFAULTS;
      cp.compcode = blosc2_compname_to_compcode(cnames[c]); cp.clevel = 5; cp.typesize = ts; cp.nthreads = 1;
      memset(cp.filters, 0, sizeof cp.filters);
      if (shuffles[s]==1) cp.filters[BLOSC2_MAX_FILTERS-1] = BLOSC_SHUFFLE;
      if (shuffles[s]==2) cp.filters[BLOSC2_MAX_FILTERS-1] = BLOSC_BITSHUFFLE;
      blosc2_context *ctx = blosc2_create_cctx(cp);
      blosc2_ctx_set_blosc1_format(ctx, true);
      cb = blosc2_compress_ctx(ctx, src, (int32_t)nb, dst, (int32_t)(nb + BLOSC2_MAX_OVERHEAD));
      blosc2_free_ctx(ctx);
    } else {
      blosc1_set_compressor(cnames[c]);
      cb = blosc1_compress(5, shuffles[s], ts, nb, src, dst, nb + BLOSC2_MAX_OVERHEAD);
    }
    if (cb <= 0) { fprintf(stderr, "compress fail %s s%d t%d n%zu: %d\n", cnames[c], s, t, nb, cb); return 2; }
    char fn[256]; snprintf(fn, sizeof fn, "out/%s_%s_s%d_t%d_n%zu.bin", mode, cnames[c], s, ts, nb);
    FILE *f = fopen(fn, "wb"); fwrite(dst, 1, cb, f); fclose(f); n++;
    free(src); free(dst);
  }
  printf("%s: wrote %d buffers; first byte(version) of last = see check\n", mode, n);
  return 0;
}
