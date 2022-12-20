#include <stdint.h>
#include <stddef.h>

// https://github.com/Unidata/netcdf-c/blob/8eb71290eb9360dcfd4955ba94759ba8d02c40a9/plugins/H5checksum.c


uint32_t H5_checksum_fletcher32(const void *_data, size_t _len)
{
    const uint8_t *data = (const uint8_t *)_data;  /* Pointer to the data to be summed */
    size_t len = _len / 2;      /* Length in 16-bit words */
    uint32_t sum1 = 0, sum2 = 0;


    /* Compute checksum for pairs of bytes */
    /* (the magic "360" value is is the largest number of sums that can be
     *  performed without numeric overflow)
     */
    while (len) {
        size_t tlen = len > 360 ? 360 : len;
        len -= tlen;
        do {
            sum1 += (uint32_t)(((uint16_t)data[0]) << 8) | ((uint16_t)data[1]);
            data += 2;
            sum2 += sum1;
        } while (--tlen);
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    /* Check for odd # of bytes */
    if(_len % 2) {
        sum1 += (uint32_t)(((uint16_t)*data) << 8);
        sum2 += sum1;
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    } /* end if */

    /* Second reduction step to reduce sums to 16 bits */
    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);

    return (sum2 << 16) | sum1;
} /* end H5_checksum_fletcher32() */
