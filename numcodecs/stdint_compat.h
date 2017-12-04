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


//static inline void store_le64(char *c, int y)
//{
//    uint64_t x = (uint64_t) y;
//    c[0] = x & 0xff;
//    c[1] = (x >> 8) & 0xff;
//    c[2] = (x >> 16) & 0xff;
//    c[3] = (x >> 24) & 0xff;
//    c[4] = (x >> 32) & 0xff;
//    c[5] = (x >> 40) & 0xff;
//    c[6] = (x >> 48) & 0xff;
//    c[7] = (x >> 56) & 0xff;
//}
//
//
//static inline int load_le64(const char *c)
//{
//    const uint8_t *d = (const uint8_t *) c;
//    uint64_t x = (
//        d[0] |
//        (d[1] << 8) |
//        (d[2] << 16) |
//        (d[3] << 24) |
//        (d[4] << 32) |
//        (d[5] << 40) |
//        (d[6] << 48) |
//        (d[7] << 56)
//    );
//    return (int) x;
//}


#ifdef inline
#undef inline
#endif


static const int UINT32_SIZE = sizeof (uint32_t);
static const int UINT64_SIZE = sizeof (uint64_t);
