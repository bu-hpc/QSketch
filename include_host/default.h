#pragma once

namespace qsketch {
    namespace default_values {
        /* 
            device_buf_sz: 
            the program will copy a batch of elements to the device,
        */
        // extern const size_t device_buf_sz;
        // extern const size_t seed_sz;

        // extern const size_t hash_table_sz;
        
        // extern const uint INC_PER_INSERT;
        

        /*
            a large prime number
        */
        extern const size_t PRIME_NUMBER;
        extern const size_t NEXT_LEVEL_ID_START;
        extern const dim3 grid_dim;
        extern const dim3 block_dim;
        extern const uint WARP_SIZE;


        extern const size_t seed_sz;
        extern const size_t HASH_MASK_ONES;
        extern const size_t HASH_MASK_TABLE_SZ;

        __device__ extern const uint DEVICE_WARP_SIZE;
        // extern const size_t DEVICE_HASH_MASK_ONES;
        __device__ extern const size_t DEVICE_NEXT_LEVEL_ID_START;
    }

}

