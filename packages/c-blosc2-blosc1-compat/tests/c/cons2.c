// consumer: c-blosc1 decompress; verifies against regenerated data
#include <blosc2.h>
#include "common.h"
int main(int argc, char **argv) {
  const char *mode = argv[1];
  blosc2_init(); blosc2_set_nthreads(1);
  int ok = 0, bad = 0;
  for (int c = 0; c < 5; c++) for (int s = 0; s < 3; s++) for (int t = 0; t < 7; t++) for (int z = 0; z < 7; z++) {
    size_t nb = sizes[z]; int ts = typesizes[t];
    
    char fn[256]; snprintf(fn, sizeof fn, "out/%s_%s_s%d_t%d_n%zu.bin", mode, cnames[c], shuffles[s], ts, nb);
    FILE *f = fopen(fn, "rb"); if (!f) { printf("missing %s\n", fn); continue; }
    uint8_t *cbuf = malloc(nb + 1024); size_t cb = fread(cbuf, 1, nb + 1024, f); fclose(f);
    uint8_t *want = malloc(nb), *got = calloc(1, nb); fill(want, nb, ts, 1234);
    int r = blosc1_decompress(cbuf, got, nb);
    if (r == (int)nb && !memcmp(want, got, nb)) ok++;
    else { if (bad < 100000) printf("BAD %s ret=%d ver=%d\n", fn, r, cbuf[0]); bad++; }
    free(cbuf); free(want); free(got);
  }
  printf("%s: c-blosc1 decoded ok=%d bad=%d\n", mode, ok, bad);
  return bad != 0;
}
