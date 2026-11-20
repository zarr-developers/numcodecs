#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static const char *cnames[] = {"blosclz","lz4","lz4hc","zlib","zstd"};
static const int shuffles[] = {0,1,2};
static const int typesizes[] = {1,2,3,4,7,8,16};
static const size_t sizes[] = {1, 33, 127, 1001, 4096, 65537, 100003};
static void fill(uint8_t *p, size_t n, int ts, unsigned seed) {
  unsigned x = seed;
  for (size_t i = 0; i < n; i++) { if ((i % ts) == 0) x = x*1103515245u + 12345u; p[i] = (uint8_t)((i/ts)*3 + ((i%ts)? (x>>16)&0x7 : 0)); }
}
