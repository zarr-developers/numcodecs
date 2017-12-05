#if defined(_WIN32) && defined(_MSC_VER)
#define inline __inline
#elif defined(__SUNPRO_C) || defined(__hpux) || defined(_AIX)
#define inline
#endif


#if defined(_WIN32) && defined(_MSC_VER)
#if _MSC_VER >= 1600
#include <stdint.h>
#else /* _MSC_VER >= 1600 */
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
#endif /* _MSC_VER >= 1600 */
#endif


static inline void store_le32(char *c, int y)
{
    uint32_t x = (uint32_t) y;
    c[0] = x & 0xff;
    c[1] = (x >> 8) & 0xff;
    c[2] = (x >> 16) & 0xff;
    c[3] = (x >> 24) & 0xff;
}


static inline int load_le32(const char *c)
{
    const uint8_t *d = (const uint8_t *) c;
    uint32_t x = d[0] | (d[1] << 8) | (d[2] << 16) | (d[3] << 24);
    return (int) x;
}


#ifdef inline
#undef inline
#endif


static const int UINT32_SIZE = sizeof (uint32_t);
