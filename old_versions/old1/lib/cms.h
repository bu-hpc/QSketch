#ifndef CMS_HEADER
#define CMS_HEADER 

#include "hash.h"

template <typename Key_T, typename Hash_T>
struct CMS
{
    // Count-Min Sketch Hash Map
    size_t nfuns = 0;
    bool insert(const Key_T *key, size_t sz) {

    }
    template <typename Count_T>
    bool search(const Key_T *key, size_t sz, Count_T *c) {

    }
};


#endif