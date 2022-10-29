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


static inline void store_le32(uint8_t c[4], uint32_t i)
{
    c[0] = i & 0xFF;
    c[1] = (i >> 8) & 0xFF;
    c[2] = (i >> 16) & 0xFF;
    c[3] = (i >> 24) & 0xFF;
}


static inline uint32_t load_le32(const uint8_t c[4])
{
    return (
        c[0] |
        (c[1] << 8) |
        (c[2] << 16) |
        (c[3] << 24)
    );
}


#ifdef inline
#undef inline
#endif


static const int UINT32_SIZE = sizeof (uint32_t);
