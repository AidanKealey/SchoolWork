
#ifndef _KVRPC_H_
#define _KVRPC_H_

struct kvGetResult{
    bool status;
    uint8_t * value;
    uint16_t  vlen;
};

#endif
