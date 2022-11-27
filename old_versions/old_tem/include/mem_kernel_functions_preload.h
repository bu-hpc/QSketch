#pragma once


// __device__ inline load1() {
//     Key_T v = keys[i];
//     return v;
// }

template <typename T>
__constant__ T constant_seed[CONSTANT_SEED_BUFFER_SIZE];

template <typename T>
int copy_to_constant_seed(T *seed, size_t sz) {
    if (sz > CONSTANT_SEED_BUFFER_SIZE) {
        return -1;
    }
    cudaMemcpyToSymbol(constant_seed<T>, seed, sizeof(T) * sz);
    return 0;
} 

// cal1 fast
template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void cal1(unsigned char *hash_mask, Count_T **htl,
    
    Count_T *hash_table_low, size_t n_low,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz,
    
    size_t tid, const Key_T &v) {
    Hashed_T hv = hash(seed + tid * s_sz, s_sz, v);
    Hashed_T htl_offset;
    if (tid == 0) {
        htl_offset = (hv % n_low) * WARP_SIZE;
    }
    htl_offset = __shfl_sync(0xffffffff, htl_offset, 0);
    *htl = hash_table_low + htl_offset;

    unsigned char hash_bit = 0x0f;

    #pragma unroll
    for (int i = 0; i < 20; i += 4) {
        hash_bit &= (hv >> i);
    }

    unsigned char hash_bit_next = __shfl_down_sync(0xffffffff, hash_bit, 1);
    if (tid % 2 == 0) {
        hash_bit |= hash_bit_next << 4;
        hash_mask[tid/2] = hash_bit;
    }

    if (tid == 0) {
        Hashed_T hv_mask = hv % 124;
        hash_mask[hv_mask/8] |= 1 << (hv_mask % 8);
        hash_mask[15] &= 0b00001111;
    }
    __threadfence_block();
    // return 0;
}

// template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
// __device__ inline void cal1(unsigned char *hash_mask, Count_T **htl,
    
//     Count_T *hash_table_low, size_t n_low,
//     const Hash_Function &hash,
//     Seed_T *seed, size_t s_sz,
    
//     size_t tid, const Key_T &v) {
//     Hashed_T hv = hash(seed + tid * s_sz, s_sz, v);
//     Hashed_T htl_offset;
//     if (tid == 0) {
//         htl_offset = (hv % n_low) * WARP_SIZE;
//     }
//     htl_offset = __shfl_sync(0xffffffff, htl_offset, 0);
//     *htl = hash_table_low + htl_offset;

//     unsigned char hash_bit = 0x0f;

//     #pragma unroll
//     for (int i = 0; i < 20; i += 4) {
//         hash_bit &= (hv >> i);
//     }

//     unsigned char hash_bit_next = __shfl_down_sync(0xffffffff, hash_bit, 1);
//     if (tid % 2 == 0) {
//         hash_bit |= hash_bit_next << 4;
//         hash_mask[tid/2] = hash_bit;
//     }

//     if (tid == 0) {
//         Hashed_T hv_mask = hv % 124;
//         hash_mask[hv_mask/8] |= 1 << (hv_mask % 8);
//         hash_mask[15] &= 0b00001111;
//     }
//     __threadfence_block();
//     // return 0;
// }


// cal1 for bus width

// #define BUS_WIDTH 8

// template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
// __device__ inline void cal1(unsigned char *hash_mask, Count_T **htl,
    
//     Count_T *hash_table_low, size_t n_low,
//     const Hash_Function &hash,
//     Seed_T *seed, size_t s_sz,
    
//     size_t tid, const Key_T &v) {
//     Hashed_T hv = hash(seed + tid * s_sz, s_sz, v);
//     Hashed_T htl_offset;
//     Hashed_T sub_warp_id;
//     if (tid == 0) {
//         htl_offset = (hv % n_low) * WARP_SIZE;
//         sub_warp_id = (hv % 97) % 8;
//     }
//     htl_offset = __shfl_sync(0xffffffff, htl_offset, 0);
//     sub_warp_id = __shfl_sync(0xffffffff, sub_warp_id, 0);
//     *htl = hash_table_low + htl_offset;


//     unsigned char hash_bit = 0xff;

//     if (tid < 16) {
//         hash_mask[tid] = 0x00;
//     }

//     if (tid < 2) {
//         #pragma unroll
//         for (int i = 0; i < 16; i += 8) {
//             hash_bit &= (hv >> i);
//         }
//         hash_mask[sub_warp_id * 8 + tid] = hash_bit;
//     }
//     __threadfence_block();
//     if (tid == 0) {
//         Hashed_T hv_mask = hv % 12;
//         hash_mask[sub_warp_id * 8 + hv_mask/8] |= 1 << (hv_mask % 8);

//         hash_mask[15] &= 0b00001111;
//     }
//     __threadfence_block();
//     // return 0;
// }



template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void insert_high(Count_T *hash_table_high, const Count_T &htl_31, 
    unsigned char *hash_mask, size_t tid,
    const unsigned int &warp_mask_id, const unsigned char &warp_mask) {
    Count_T *hth = hash_table_high + (htl_31 - BUFFER_START) * 128;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (hash_mask[i * 4 + warp_mask_id] & warp_mask) {
            atomicAdd(hth + i * WARP_SIZE + tid, 1);
        }
    }
    // return 0;
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline Count_T insert_low_max(unsigned char *hash_mask, Count_T *htl, size_t tid) {
    Count_T max_count = 0;
    Count_T add = 0;
    Count_T cv = htl[tid];
    unsigned char thm = (hash_mask[tid/2]);
    thm = thm >> ((tid % 2) * 4);
    
    if (thm & 0b00000001) {
        add |= (1u);
        max_count = max(max_count, (cv & 0x000000ffu));
    }

    if (thm & 0b00000010) {
        add |= (1u << 8);
        max_count = max(max_count, (cv & 0x0000ff00u) >> 8);
    }

    if (thm & 0b00000100) {
        add |= (1u << 16);
        max_count = max(max_count, (cv & 0x00ff0000u) >> 16);
    }

    if (thm & 0b00001000) {
        add |= (1u << 24);
        max_count = max(max_count, (cv & 0xff000000u) >> 24);
    }
    
    // bottleneck
    for (int j = 16; j >= 1; j = j >> 1) {
        Count_T t_max = __shfl_down_sync(0xffffffff, max_count, j);
        if (tid < j) {
            max_count = max(max_count, t_max);
        }
    }

    max_count = __shfl_sync(0xffffffff, max_count, 0);

    // // add = ((tid + cv) % 8 == 0) ? 1 : 0;
    // // max_count = 128;

    // if (max_count > 128) {
    //     printf("update\n");
    // }

    if (add != 0 && max_count <= 128) {
        atomicAdd(htl + tid, add);
    }

    return max_count;
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline Count_T insert_low_any(unsigned char *hash_mask, Count_T *htl, size_t tid) {
    Count_T max_count = 0;
    Count_T add = 0;
    Count_T cv = htl[tid];
    unsigned char thm = (hash_mask[tid/2]);
    thm = thm >> ((tid % 2) * 4);
    
    if (thm & 0b00000001) {
        add |= (1u);
        max_count = max(max_count, (cv & 0x000000ffu));
    }

    if (thm & 0b00000010) {
        add |= (1u << 8);
        max_count = max(max_count, (cv & 0x0000ff00u) >> 8);
    }

    if (thm & 0b00000100) {
        add |= (1u << 16);
        max_count = max(max_count, (cv & 0x00ff0000u) >> 16);
    }

    if (thm & 0b00001000) {
        add |= (1u << 24);
        max_count = max(max_count, (cv & 0xff000000u) >> 24);
    }
    int update_low = __any_sync(0xffffffff, max_count > 128);

    if (add != 0 && (update_low == 0)) {
        atomicAdd(htl + tid, add);
    }

    return update_low;
}


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline unsigned int atomicMalloc(Count_T *htl, unsigned int *mem_id) {
    unsigned int id = 0;
    unsigned int old = atomicCAS(htl + 31, 0, 1);
    if (old == 0) {
        id = atomicAdd(mem_id, 1);
        htl[31] = id;
    } else {
        // while (__ldcv(htl + 31) <= 1) {}
        // tem = __ldcv(htl + 31);
        // id = __ldcv(htl + 31);
        while (id <= 1) {
            id = max(id, __ldcv(htl + 31));
            __threadfence_block();
        }
    }
    return id;
}

// #define USE_INSERT_LOW_MAX

#ifndef USE_INSERT_LOW_MAX
#define USE_INSERT_LOW_ANY
#endif

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void cal2(Count_T *hash_table_high, unsigned int *mem_id,
    Count_T *htl, Count_T htl_31, unsigned char *hash_mask, size_t tid)
{
    unsigned int warp_mask_id = tid/8;
    unsigned char warp_mask = 1u << (tid % 8);

    
    if (htl_31 > BUFFER_START) {
        insert_high<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_table_high, htl_31, hash_mask, tid, warp_mask_id, warp_mask);
    } else {
        #ifdef USE_INSERT_LOW_MAX
        Count_T max_count = insert_low_max<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_mask, htl, tid);
        if (max_count > 128) {
            unsigned int id = 0;
            if (tid == 0) {
                // printf("update_low\n");
                id = atomicMalloc<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(htl, mem_id);
            }
            __threadfence_block();
            id = __shfl_sync(0xffffffff, id, 0);
            insert_high<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_table_high, id, hash_mask, tid, warp_mask_id, warp_mask);
        }
        #endif

        #ifdef USE_INSERT_LOW_ANY
        Count_T update_low = insert_low_any<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_mask, htl, tid);
        if (update_low) {
            unsigned int id = 0;
            if (tid == 0) {
                // printf("update_low\n");
                id = atomicMalloc<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(htl, mem_id);
            }
            __threadfence_block();
            id = __shfl_sync(0xffffffff, id, 0);
            insert_high<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_table_high, id, hash_mask, tid, warp_mask_id, warp_mask);
        }

        #endif
    }
} 


// insert_preload_fast
template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_preload(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }

    Key_T preload_key;
    if (b < sz) {
        preload_key = keys[b];
    }

    if ((e - b) % 2 == 1) {
        size_t i = b + 1;
        __shared__ unsigned char result_1_hash_mask[16];
        __shared__ unsigned char result_3_hash_mask[16];

            // /*
            //     test the workload of hash_mask
            // */

            //     if (tid < 16) {
            //         result_1_hash_mask[tid] = 0;
            //         if (tid % 4 == 0) {
            //             result_1_hash_mask[tid] = (1 << tid);
            //         }
            //     }

            //     if (tid < 16) {
            //         result_3_hash_mask[tid] = 0;
            //         if (tid % 4 == 0) {
            //             result_3_hash_mask[tid] = (1 << tid);
            //         }
            //     }
            //     __threadfence_block();
            // /*
            //     end
            // */


        for (; i < e; i += 2) {
            Key_T result_0_key = keys[i];
            
            
            Count_T *result_1_htl;
            cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                result_1_hash_mask, &result_1_htl,
                hash_table_low, n_low,
                hash,
                seed, s_sz,
                tid, preload_key);

            Count_T result_2_htl_31 = result_1_htl[31];

            
            Count_T *result_3_htl;
            cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                result_3_hash_mask, &result_3_htl,
                hash_table_low, n_low,
                hash,
                seed, s_sz,
                tid, result_0_key);

            Count_T result_4_htl_31 = result_3_htl[31];

            cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, mem_id,
                result_1_htl, result_2_htl_31, result_1_hash_mask, tid);

            preload_key = keys[i + 1];

            cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, mem_id,
                result_3_htl, result_4_htl_31, result_3_hash_mask, tid);           

        }

        // __shared__ unsigned char result_1_hash_mask[16];
        Count_T *result_1_htl;
        cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            result_1_hash_mask, &result_1_htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, preload_key);
        
        Count_T result_2_htl_31 = result_1_htl[31];

        cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_table_high, mem_id,
            result_1_htl, result_2_htl_31, result_1_hash_mask, tid);



    } else {
        size_t i = b + 1;
        __shared__ unsigned char result_1_hash_mask[16];
        __shared__ unsigned char result_3_hash_mask[16];

            // /*
            //     test the workload of hash_mask
            // */

            //     if (tid < 16) {
            //         result_1_hash_mask[tid] = 0;
            //         if (tid % 4 == 0) {
            //             result_1_hash_mask[tid] = (1 << tid);
            //         }
            //     }

            //     if (tid < 16) {
            //         result_3_hash_mask[tid] = 0;
            //         if (tid % 4 == 0) {
            //             result_3_hash_mask[tid] = (1 << tid);
            //         }
            //     }
            //     __threadfence_block();
            // /*
            //     end
            // */
        for (; i < e - 2; i += 2) {
            Key_T result_0_key = keys[i];
            
            // __shared__ unsigned char result_1_hash_mask[16];
            Count_T *result_1_htl;
            cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                result_1_hash_mask, &result_1_htl,
                hash_table_low, n_low,
                hash,
                seed, s_sz,
                tid, preload_key);

            Count_T result_2_htl_31 = result_1_htl[31];

            // __shared__ unsigned char result_3_hash_mask[16];
            Count_T *result_3_htl;
            cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                result_3_hash_mask, &result_3_htl,
                hash_table_low, n_low,
                hash,
                seed, s_sz,
                tid, result_0_key);

            Count_T result_4_htl_31 = result_3_htl[31];

            cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, mem_id,
                result_1_htl, result_2_htl_31, result_1_hash_mask, tid);

            preload_key = keys[i + 1];

            cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, mem_id,
                result_3_htl, result_4_htl_31, result_3_hash_mask, tid);    
        }

        Key_T result_0_key = keys[i];
            
        // __shared__ unsigned char result_1_hash_mask[16];
        Count_T *result_1_htl;
        cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            result_1_hash_mask, &result_1_htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, preload_key);

        Count_T result_2_htl_31 = result_1_htl[31];

        // __shared__ unsigned char result_3_hash_mask[16];
        Count_T *result_3_htl;
        cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            result_3_hash_mask, &result_3_htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, result_0_key);

        Count_T result_4_htl_31 = result_3_htl[31];

        cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_table_high, mem_id,
            result_1_htl, result_2_htl_31, result_1_hash_mask, tid);

        cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_table_high, mem_id,
            result_3_htl, result_4_htl_31, result_3_hash_mask, tid);    

    }  
}



// template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
// __global__ void insert_warp_mem_threadfence_preload(Key_T *keys, size_t sz,
//     Count_T *hash_table_low, size_t n_low,
//     Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
//     const Hash_Function &hash,
//     Seed_T *seed, size_t s_sz, 
//     size_t work_load_per_warp,
//     void *debug)
// {
//     size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
//     size_t tid = threadIdx.x % WARP_SIZE;
//     size_t b = wid * work_load_per_warp;
//     size_t e = (wid + 1) * work_load_per_warp;

//     if (e >= sz) {
//         e = sz;
//     }

//     Key_T preload_key;
//     if (b < sz) {
//         preload_key = keys[b];
//     }

//     if ((e - b) % 2 == 1) {
//         size_t i = b + 1;
//         for (; i < e; i += 2) {
//             Key_T result_0_key = keys[i];
            
//             __shared__ unsigned char result_1_hash_mask[16];
//             Count_T *result_1_htl;
//             cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
//                 result_1_hash_mask, &result_1_htl,
//                 hash_table_low, n_low,
//                 hash,
//                 seed, s_sz,
//                 tid, preload_key);

//             Count_T result_2_htl_31 = result_1_htl[31];

//             __shared__ unsigned char result_3_hash_mask[16];
//             Count_T *result_3_htl;
//             cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
//                 result_3_hash_mask, &result_3_htl,
//                 hash_table_low, n_low,
//                 hash,
//                 seed, s_sz,
//                 tid, result_0_key);

//             Count_T result_4_htl_31 = result_3_htl[31];

//             cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
//                 hash_table_high, mem_id,
//                 result_1_htl, result_2_htl_31, result_1_hash_mask, tid);

//             preload_key = keys[i + 1];

//             cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
//                 hash_table_high, mem_id,
//                 result_3_htl, result_4_htl_31, result_3_hash_mask, tid);           

//         }

//         __shared__ unsigned char result_1_hash_mask[16];
//         Count_T *result_1_htl;
//         cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
//             result_1_hash_mask, &result_1_htl,
//             hash_table_low, n_low,
//             hash,
//             seed, s_sz,
//             tid, preload_key);
        
//         Count_T result_2_htl_31 = result_1_htl[31];

//         cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
//             hash_table_high, mem_id,
//             result_1_htl, result_2_htl_31, result_1_hash_mask, tid);



//     } else {
//         size_t i = b + 1;
//         for (; i < e - 2; i += 2) {
//             Key_T result_0_key = keys[i];
            
//             __shared__ unsigned char result_1_hash_mask[16];
//             Count_T *result_1_htl;
//             cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
//                 result_1_hash_mask, &result_1_htl,
//                 hash_table_low, n_low,
//                 hash,
//                 seed, s_sz,
//                 tid, preload_key);

//             Count_T result_2_htl_31 = result_1_htl[31];

//             __shared__ unsigned char result_3_hash_mask[16];
//             Count_T *result_3_htl;
//             cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
//                 result_3_hash_mask, &result_3_htl,
//                 hash_table_low, n_low,
//                 hash,
//                 seed, s_sz,
//                 tid, result_0_key);

//             Count_T result_4_htl_31 = result_3_htl[31];

//             cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
//                 hash_table_high, mem_id,
//                 result_1_htl, result_2_htl_31, result_1_hash_mask, tid);

//             preload_key = keys[i + 1];

//             cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
//                 hash_table_high, mem_id,
//                 result_3_htl, result_4_htl_31, result_3_hash_mask, tid);    
//         }

//         Key_T result_0_key = keys[i];
            
//         __shared__ unsigned char result_1_hash_mask[16];
//         Count_T *result_1_htl;
//         cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
//             result_1_hash_mask, &result_1_htl,
//             hash_table_low, n_low,
//             hash,
//             seed, s_sz,
//             tid, preload_key);

//         Count_T result_2_htl_31 = result_1_htl[31];

//         __shared__ unsigned char result_3_hash_mask[16];
//         Count_T *result_3_htl;
//         cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
//             result_3_hash_mask, &result_3_htl,
//             hash_table_low, n_low,
//             hash,
//             seed, s_sz,
//             tid, result_0_key);

//         Count_T result_4_htl_31 = result_3_htl[31];

//         cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
//             hash_table_high, mem_id,
//             result_1_htl, result_2_htl_31, result_1_hash_mask, tid);

//         cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
//             hash_table_high, mem_id,
//             result_3_htl, result_4_htl_31, result_3_hash_mask, tid);    

//     }  
// }

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_without_preload(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }
    
    __shared__ unsigned char hash_mask[16];

    /*
        test the workload of hash_mask
    */

        if (tid < 16) {
            hash_mask[tid] = 0;
            if (tid % 4 == 0) {
                hash_mask[tid] = (1 << tid);
            }
        }
        __threadfence_block();
    /*
        end
    */

    for (size_t i = b; i < e; ++i) {

        Key_T v = keys[i]; // load 1
        Count_T *htl;

        cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_mask, &htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, v);
        // htl = hash_table_low + keys[i] % n_low;
        
        Count_T htl_31 = htl[31];
        cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_table_high, mem_id,
            htl, htl_31, hash_mask, tid);
        // if ( (keys[i] + tid) % 8 == 0)
            // atomicAdd(htl + 31, 1);
    }    
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_example1(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }
    
    // __shared__ unsigned char hash_mask[16];
    // Key_T v = keys[0];

    // __shared__ Seed_T shared_seed[32];
    // for (size_t i = tid; i < s_sz && i < 32; i += WARP_SIZE) {
    //     shared_seed[i] = seed[i];
    // }

    // Seed_T sv = seed[tid];

    for (size_t i = b; i < e; ++i) {

        // v = v * 8388593;
        Key_T v = keys[i];
        Count_T *htl;


        Hashed_T hv = hash(seed + tid * s_sz, s_sz, v);
        // Hashed_T hv = hash(constant_seed<Seed_T> + tid * s_sz, s_sz, v);
        // Hashed_T hv = v;
        // #pragma unroll
        // for (int j = 0; j < 4; ++j) {
            // hv = (hv * shared_seed[tid]) + (65521); 
            // hv = (hv * sv) + (65521);
        // }
        // Hashed_T hv = (v * 8388593) * (65521); 
        // Hashed_T hv = hash(seed, s_sz, v);

        htl = hash_table_low + hv % n_low;

        // htl = hash_table_low + (v * 8388593) % n_low;

        if (htl[31]) {
            if ( (hv + tid) % 2 == 0)
                atomicAdd(htl + tid, 1);
        } else {
            if ( (hv + tid) % 8 == 0)
                atomicAdd(htl + tid, 1);
        }      
    }    
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_example2(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }

    for (size_t i = b + tid; i < e; i += WARP_SIZE) {
        Key_T v = keys[i];
        Count_T *htl = hash_table_low + keys[i] % n_low;
        atomicAdd(htl + 31, 1);
        // atomicAdd(hash_table_low + (i % n_low), 1);
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_example3_1(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }
    
    for (size_t i = b; i < e; ++i) {

        Key_T v = keys[i];

        Hashed_T hv = (v * 8388593) * (65521); 
        // Hashed_T hv = hash(seed, s_sz, v);

        Count_T *htl = hash_table_low + hv % n_low;

        if ( (v + tid) % 16 == 0)
            atomicAdd(htl + tid, 1);

        // if (tid < 16 && (keys[i] + tid) % 4 == 0)
        //     atomicAdd(htl + tid, 1);

        // if (tid < 8 && (keys[i] + tid) % 2 == 0)
        //     atomicAdd(htl + tid, 1); 
    }    
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_example3_2(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }
    
    for (size_t i = b; i < e; ++i) {

        Key_T v = keys[i];

        Hashed_T hv = (v * 8388593) * (65521); 
        // Hashed_T hv = hash(seed, s_sz, v);

        Count_T *htl = hash_table_low + hv % n_low;

        // if ( (keys[i] + tid) % 8 == 0)
        //     atomicAdd(htl + tid, 1);

        if (tid < 16 && (v + tid) % 8 == 0)
            atomicAdd(htl + tid, 1);

        // if (tid < 8 && (keys[i] + tid) % 2 == 0)
        //     atomicAdd(htl + tid, 1); 
    }    
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_example3_3(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }
    
    for (size_t i = b; i < e; ++i) {

        Key_T v = keys[i];

        Hashed_T hv = (v * 8388593) * (65521); 
        // Hashed_T hv = hash(seed, s_sz, v);

        Count_T *htl = hash_table_low + hv % n_low;

        // if ( (keys[i] + tid) % 8 == 0)
        //     atomicAdd(htl + tid, 1);

        // if (tid < 16 && (keys[i] + tid) % 4 == 0)
        //     atomicAdd(htl + tid, 1);

        if (tid < 8 && (v + tid) % 4 == 0)
            atomicAdd(htl + tid, 1); 
    }    
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_example4_1(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }

    __shared__ Hashed_T shv[WARP_SIZE];
    for (size_t i = b + tid; i < e; i += WARP_SIZE) {
        Key_T v = keys[i];
        // Hashed_T hv = example_4_hash()
        Hashed_T hv = 0;
        #pragma unroll
        for (size_t j = 0; j < s_sz; ++j) {
            hv += v * constant_seed<Seed_T>[j];
        }

        shv[tid] = hv;

        #pragma unroll
        for (size_t k = 0; k < WARP_SIZE; ++k) {
            Count_T *htl;
            htl = hash_table_low + shv[k] % n_low;

            // htl = hash_table_low + (v * 8388593) % n_low;

            if (htl[31]) {
                if ( (shv[k] + tid) % 2 == 0)
                    atomicAdd(htl + tid, 1);
            } else {
                if ( (shv[k] + tid) % 8 == 0)
                    atomicAdd(htl + tid, 1);
            }  
        }
    }   
}


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_example4_2(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }

    __shared__ Hashed_T shv[WARP_SIZE][WARP_SIZE];
    for (size_t i = b + tid; i < e; i += WARP_SIZE) {
        Key_T v = keys[i];
        // Hashed_T hv = example_4_hash()
        // Hashed_T hv = 0;
        // #pragma unroll
        // for (size_t j = 0; j < s_sz; ++j) {
        //     hv += v * seed[j];
        // }

        #pragma unroll
        for (size_t j = 0; j < WARP_SIZE; ++j)
        {
            // Hashed_T hv = hash(seed + j * s_sz, s_sz, v);
            Hashed_T hv = 0;
            #pragma unroll
            for (size_t k = 0; k < s_sz; ++k) {
                hv += v * constant_seed<Seed_T>[j * s_sz + k];
                // hv += v * seed[j * s_sz + k];
            }
            shv[j][tid] = hv;
        }

        

        #pragma unroll
        for (size_t k = 0; k < WARP_SIZE; ++k) {
            Count_T *htl;
            htl = hash_table_low + shv[k][tid] % n_low;

            // htl = hash_table_low + (v * 8388593) % n_low;

            if (htl[31]) {
                if ( (shv[k][tid] + tid) % 2 == 0)
                    atomicAdd(htl + tid, 1);
            } else {
                if ( (shv[k][tid] + tid) % 8 == 0)
                    atomicAdd(htl + tid, 1);
            }  
        }
    }   
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_example5_1(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }

    for (size_t i = b + tid; i < e; i += WARP_SIZE) {
        Key_T v = keys[i];
        Hashed_T hv0 = hash(seed, s_sz, v);
        size_t iv0 = hv0 % (n_low * WARP_SIZE);
        Hashed_T hv1 = hash(seed + 1, s_sz, v);
        size_t iv1 = hv1 % (n_low * WARP_SIZE);
        Hashed_T hv2 = hash(seed + 2, s_sz, v);
        size_t iv2 = hv2 % (n_low * WARP_SIZE);

        // size_t iv1 = iv0 + 3;
        // size_t iv2 = iv0 + 6;

        // if (iv0 >= n_low * WARP_SIZE) {
        //     printf("err\n");
        // }

        atomicAdd(hash_table_low + iv0, 1);
        atomicAdd(hash_table_low + iv1, 1);
        atomicAdd(hash_table_low + iv2, 1);
    }   
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_warp_mem_threadfence_example5_1(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    Count_T *count, Count_T Count_MAX,
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }


    for (size_t i = b + tid; i < e; i += WARP_SIZE) {
        Key_T v = keys[i];
        Hashed_T hv0 = hash(seed, s_sz, v);
        size_t iv0 = hv0 % (n_low * WARP_SIZE);
        // Hashed_T hv1 = hash(seed + 1, s_sz, v);
        // size_t iv1 = hv1 % (n_low * WARP_SIZE);
        // Hashed_T hv2 = hash(seed + 2, s_sz, v);
        // size_t iv2 = hv2 % (n_low * WARP_SIZE);

        size_t iv1 = iv0 + 3;
        size_t iv2 = iv0 + 6;

        Count_T a = 0xffffffff;
        a = min(a, hash_table_low[iv0]);
        a = min(a, hash_table_low[iv1]);
        a = min(a, hash_table_low[iv2]);
        count[i] = a;
    }  
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_example5_2(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }

    for (size_t j = 1; j <= 32; ++j) {
        for (size_t i = b + tid; i < e; i += WARP_SIZE) {
            Key_T v = keys[i];
            Hashed_T hv0 = hash(seed, s_sz, v);
            size_t iv0 = hv0 % (n_low * WARP_SIZE);
            Hashed_T hv1 = hash(seed + 1, s_sz, v);
            size_t iv1 = hv1 % (n_low * WARP_SIZE);
            Hashed_T hv2 = hash(seed + 2, s_sz, v);
            size_t iv2 = hv2 % (n_low * WARP_SIZE);

            size_t t0 = iv0 / (4 * 1024 * 1024);
            if (t0 >= j - 1 && t0 < j) {
                atomicAdd(hash_table_low + iv0, 1);    
            }
            size_t t1 = iv1 / (4 * 1024 * 1024);
            if (t1 >= j - 1 && t1 < j) {
                atomicAdd(hash_table_low + iv1, 1);    
            }
            size_t t2 = iv2 / (4 * 1024 * 1024);
            if (t2 >= j - 1 && t2 < j) {
                atomicAdd(hash_table_low + iv2, 1);    
            }
        } 
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_example5_3(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }

    size_t work_load_per_thread = (e > b + tid) ? (e - b - tid) : 0;
    work_load_per_thread = work_load_per_thread / WARP_SIZE;
    // size_t *jptr = static_cast<size_t *>(debug);
    size_t j = n_high;
    for (size_t i = b + tid; i < e; i += WARP_SIZE) {
        Key_T v = keys[i];
        Hashed_T hv0 = hash(seed, s_sz, v);
        size_t iv0 = hv0 % (n_low * WARP_SIZE);
        Hashed_T hv1 = hash(seed + 1, s_sz, v);
        size_t iv1 = hv1 % (n_low * WARP_SIZE);
        Hashed_T hv2 = hash(seed + 2, s_sz, v);
        size_t iv2 = hv2 % (n_low * WARP_SIZE);

        size_t t0 = iv0 / (4 * 1024 * 1024);
        if (t0 >= j - 1 && t0 < j) {
            atomicAdd(hash_table_low + iv0, 1);    
        }
        size_t t1 = iv1 / (4 * 1024 * 1024);
        if (t1 >= j - 1 && t1 < j) {
            atomicAdd(hash_table_low + iv1, 1);    
        }
        size_t t2 = iv2 / (4 * 1024 * 1024);
        if (t2 >= j - 1 && t2 < j) {
            atomicAdd(hash_table_low + iv2, 1);    
        }
    }

    // while (work_load_per_thread > 0) {
    //     for (size_t i = b + tid; i < e; i += WARP_SIZE) {
    //         Key_T v = keys[i];
    //         Hashed_T hv0 = hash(seed, s_sz, v);
    //         size_t iv0 = hv0 % (n_low * WARP_SIZE);
    //         Hashed_T hv1 = hash(seed + 1, s_sz, v);
    //         size_t iv1 = hv1 % (n_low * WARP_SIZE);
    //         Hashed_T hv2 = hash(seed + 2, s_sz, v);
    //         size_t iv2 = hv2 % (n_low * WARP_SIZE);

    //         size_t t0 = iv0 / (4 * 1024 * 1024);
    //         if (t0 >= j - 1 && t0 < j) {
    //             atomicAdd(hash_table_low + iv0, 1);    
    //         }
    //         size_t t1 = iv1 / (4 * 1024 * 1024);
    //         if (t1 >= j - 1 && t1 < j) {
    //             atomicAdd(hash_table_low + iv1, 1);    
    //         }
    //         size_t t2 = iv2 / (4 * 1024 * 1024);
    //         if (t2 >= j - 1 && t2 < j) {
    //             atomicAdd(hash_table_low + iv2, 1);    
    //         }
    //     }
    // }

    // for (size_t j = 1; j <= 32; ++j) {
    //     for (size_t i = b + tid; i < e; i += WARP_SIZE) {
    //         Key_T v = keys[i];
    //         Hashed_T hv0 = hash(seed, s_sz, v);
    //         size_t iv0 = hv0 % (n_low * WARP_SIZE);
    //         Hashed_T hv1 = hash(seed + 1, s_sz, v);
    //         size_t iv1 = hv1 % (n_low * WARP_SIZE);
    //         Hashed_T hv2 = hash(seed + 2, s_sz, v);
    //         size_t iv2 = hv2 % (n_low * WARP_SIZE);

    //         size_t t0 = iv0 / (4 * 1024 * 1024);
    //         if (t0 >= j - 1 && t0 < j) {
    //             atomicAdd(hash_table_low + iv0, 1);    
    //         }
    //         size_t t1 = iv1 / (4 * 1024 * 1024);
    //         if (t1 >= j - 1 && t1 < j) {
    //             atomicAdd(hash_table_low + iv1, 1);    
    //         }
    //         size_t t2 = iv2 / (4 * 1024 * 1024);
    //         if (t2 >= j - 1 && t2 < j) {
    //             atomicAdd(hash_table_low + iv2, 1);    
    //         }
    //     }
    // }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_warp_mem_threadfence_example5_2(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    Count_T *count, Count_T Count_MAX,
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }


    for (size_t i = b + tid; i < e; i += WARP_SIZE) {
        Key_T v = keys[i];
        Hashed_T hv0 = hash(seed, s_sz, v);
        size_t iv0 = hv0 % (n_low * WARP_SIZE);
        Hashed_T hv1 = hash(seed + 1, s_sz, v);
        size_t iv1 = hv1 % (n_low * WARP_SIZE);
        Hashed_T hv2 = hash(seed + 2, s_sz, v);
        size_t iv2 = hv2 % (n_low * WARP_SIZE);

        // size_t iv1 = iv0 + 3;
        // size_t iv2 = iv0 + 6;
        
        Count_T a = 0xffffffff;
        a = min(a, hash_table_low[iv0]);
        a = min(a, hash_table_low[iv1]);
        a = min(a, hash_table_low[iv2]);
        count[i] = a;
    }  
}