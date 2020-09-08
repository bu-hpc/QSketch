#include <iostream>
#include "lib.h"

#ifdef CPU_FLAG

#include "cpu/cpu_hash_functions/hash_functions_list.h"

#endif

#ifdef GPU_FLAG

#endif

template <typename T>
struct CMS_Hash_Map
{
    // Count-Min Sketch Hash Map
    size_t nfuns = 0;

    virtual bool insert(const T*) = 0;
    virtual bool search(const T*, size_t *) = 0;
};

int main(int argc, char const *argv[])
{
    
    return 0;
}