#include <qsketch.h>


// namespace qsketch {

// extern const Mode mode = Mode::DEBUG;
// extern const bool RANDOM_SEED = true;

//     namespace default_values {
//         constexpr size_t seed_buf_sz = 128;
//         char seed_buf_local[seed_buf_sz];
//         char *seed_buf = seed_buf_local;
//         const size_t device_buf_sz = 1024 * 1024;
//         const uint INC_PER_INSERT = 3;
//         const size_t PRIME_NUMBER = 4294967291;
//         const uint WARP_SIZE = 32;
//         const size_t seed_sz = 1;

//         const size_t hash_table_sz = 1024;


//         const dim3 grid_dim(65536);
//         // const dim3 default_grid_dim(1);
//         const dim3 block_dim(32);
//     }

// }

namespace qsketch {
    namespace default_values {
        /* 
            device_buf_sz: 
            the program will copy a batch of elements to the device,
        */
        /*
            a large prime number
        */
        // const size_t PRIME_NUMBER = 4294967291;
        const size_t PRIME_NUMBER = 33554393;


        /*
            the values less than START are reserved. 
        */
        const size_t NEXT_LEVEL_ID_START = 1024;
        const dim3 grid_dim(65536);
        // const dim3 grid_dim(128);
        // const dim3 default_grid_dim(1);
        const dim3 block_dim(32);
        const uint WARP_SIZE = 32;

        const size_t seed_sz = 1;
        const size_t HASH_MASK_ONES = 3;
        const size_t HASH_MASK_TABLE_SZ = 1021;

        // __device__ const uint DEVICE_WARP_SIZE = 32;
        __device__ const uint DEVICE_WARP_SIZE = WARP_SIZE;
        // __device__ const size_t DEVICE_HASH_MASK_ONES = HASH_MASK_ONES;
        __device__ const size_t DEVICE_NEXT_LEVEL_ID_START = NEXT_LEVEL_ID_START;

    }

}
