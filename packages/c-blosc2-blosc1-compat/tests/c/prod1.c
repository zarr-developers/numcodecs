#include <blosc.h>
#include "common.h"
int main(void){ blosc_init(); blosc_set_nthreads(1);
 for (int c=0;c<5;c++) for(int s=0;s<3;s++) for(int t=0;t<7;t++) for(int z=0;z<7;z++){
  size_t nb=sizes[z]; int ts=typesizes[t]; 
  uint8_t *src=malloc(nb),*dst=malloc(nb+BLOSC_MAX_OVERHEAD); fill(src,nb,ts,1234);
  blosc_set_compressor(cnames[c]); int cb=blosc_compress(5,shuffles[s],ts,nb,src,dst,nb+BLOSC_MAX_OVERHEAD);
  if(cb<=0){fprintf(stderr,"fail\n");return 2;}
  char fn[256]; snprintf(fn,sizeof fn,"out/b1_%s_s%d_t%d_n%zu.bin",cnames[c],shuffles[s],ts,nb);
  FILE*f=fopen(fn,"wb"); fwrite(dst,1,cb,f); fclose(f);}
 return 0;}
