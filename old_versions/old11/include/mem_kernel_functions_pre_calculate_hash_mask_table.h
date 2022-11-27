#pragma once

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void pre_cal_kernel(Key_T *keys, size_t sz,
    unsigned char *hash_mask_table,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug) {
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }

    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("hello world\n");
    // }
    // size_t *debug_c = (size_t *)debug;
    // Count_T *debug_count = (Count_T *)debug;
    for (size_t i = b; i < e; ++i) {
        // if (tid == 0)
        //     atomicAdd(debug_count, 1);

        Key_T v = keys[i]; // load 1
        Hashed_T hv = hash(seed + tid * s_sz, s_sz, v);
        unsigned char *hash_mask = hash_mask_table + i * 16;

        unsigned char hash_bit = 0x0f;

        #pragma unroll
        for (int j = 0; j < 20; j += 4) {
            hash_bit &= (hv >> j);
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

        // check

        
        // if (tid == 0) {
        //     size_t c = 0;
        //     for (size_t j = 0; j < 16; ++j) {
        //         unsigned char hm = hash_mask[j];
        //         while (hm != 0) {
        //             if (hm & 1) {
        //                 c++;
        //             }
        //             hm = hm >> 1;
        //         }
        //     }
        //     if (c == 0) {
        //         printf("err c\n");
        //     }
        //     // printf("%lu\n", c);
        //     debug_c[i] = c;
        // }
    }
}


// cal1_pre_cal fast
template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void cal1_pre_cal(size_t &hash_mask_id, Count_T **htl,
    
    Count_T *hash_table_low, size_t n_low,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz,
    
    size_t tid, const Key_T &v) {
    Hashed_T hv = hash(seed + tid * s_sz, s_sz, v);
    Hashed_T htl_offset;
    if (tid == 0) {
        htl_offset = (hv % n_low) * WARP_SIZE;
        hash_mask_id = (hv % HASH_MASK_TABLE_SIZE) * HASH_MASK_SIZE;
    }
    htl_offset = __shfl_sync(0xffffffff, htl_offset, 0);
    hash_mask_id = __shfl_sync(0xffffffff, hash_mask_id, 0);
    
    *htl = hash_table_low + htl_offset;

}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void cal2_pre_call(Count_T *hash_table_high, unsigned int *mem_id,
    Count_T *htl, Count_T htl_31, unsigned char *hash_mask, size_t tid)
{
    unsigned int warp_mask_id = tid/8;
    unsigned char warp_mask = 1u << (tid % 8);

    
    if (htl_31 > BUFFER_START) {
        // if (tid == 0) {
        //     printf("called\n");
        // }
        insert_high<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_table_high, htl_31, hash_mask, tid, warp_mask_id, warp_mask);
    } else {
        // if (tid == 0) {
        //     // printf("called\n");
        // }

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

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_without_preload_pre_cal(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    unsigned char *hash_mask_table,
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

    for (size_t i = b; i < e; ++i) {

        Key_T v = keys[i]; // load 1
        

        Count_T *htl;
        size_t hash_mask_id;

        cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_mask_id, &htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, v);
        // htl = hash_table_low + keys[i] % n_low;
        
        Count_T htl_31 = htl[31];
        if (tid < 16) {
            hash_mask[tid] = hash_mask_table[hash_mask_id + tid];
            // hash_mask[tid] = hash_mask_table[(i % 1024) * 16 + tid];
            // hash_mask[tid] = hash_mask_table[(i) * 16 + tid];
            // hash_mask[tid] = 0b00001000;
        }
        // __threadfence_block();

        cal2_pre_call<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_table_high, mem_id,
            htl, htl_31, hash_mask, tid);

        // cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
        //     hash_table_high, mem_id,
        //     htl, htl_31, hash_mask, tid);
    }    
}



template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_warp_mem_threadfence_without_preload_pre_cal(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    unsigned char *hash_mask_table,
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
    
    __shared__ unsigned char hash_mask[16];

    for (size_t i = b; i < e; ++i) {
        Key_T v = keys[i]; // load 1


        Count_T *htl;
        size_t hash_mask_id;

        // search_calculation_1
        cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_mask_id, &htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, v);

        // Count_T htl_31 = htl[31];
        Count_T cv = htl[tid]; // load 2
        Count_T htl_31 = htl[31];
        if (tid < 16) {
            hash_mask[tid] = hash_mask_table[hash_mask_id + tid];
            // hash_mask[tid] = hash_mask_table[(i % 1024) * 16 + tid];
            // hash_mask[tid] = hash_mask_table[(i) * 16 + tid];
            // hash_mask[tid] = 0b00001000;
        }

        // search calculation 2
        search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_table_high, count + i, cv, htl_31, hash_mask, tid);
    }
}



template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_preload_pre_cal(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    unsigned char *hash_mask_table,
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
        __shared__ unsigned char result_2_hash_mask[16];
        __shared__ unsigned char result_4_hash_mask[16];

        for (; i < e; i += 2) {
            Key_T result_0_key = keys[i];
            
            
            Count_T *result_1_htl;
            size_t result_1_hash_mask_id;
            cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                result_1_hash_mask_id, &result_1_htl,
                hash_table_low, n_low,
                hash,
                seed, s_sz,
                tid, preload_key);

            Count_T result_2_htl_31 = result_1_htl[31];
            if (tid < 16) {
                result_2_hash_mask[tid] = hash_mask_table[result_1_hash_mask_id + tid];
            }
            
            Count_T *result_3_htl;
            size_t result_3_hash_mask_id;
            cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                result_3_hash_mask_id, &result_3_htl,
                hash_table_low, n_low,
                hash,
                seed, s_sz,
                tid, result_0_key);

            Count_T result_4_htl_31 = result_3_htl[31];
            if (tid < 16) {
                result_4_hash_mask[tid] = hash_mask_table[result_3_hash_mask_id + tid];
            }

            cal2_pre_call<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, mem_id,
                result_1_htl, result_2_htl_31, result_2_hash_mask, tid);

            preload_key = keys[i + 1];

            cal2_pre_call<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, mem_id,
                result_3_htl, result_4_htl_31, result_4_hash_mask, tid);           

        }

        Count_T *result_1_htl;
        size_t result_1_hash_mask_id;
        cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            result_1_hash_mask_id, &result_1_htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, preload_key);
        
        Count_T result_2_htl_31 = result_1_htl[31];
        if (tid < 16) {
            result_2_hash_mask[tid] = hash_mask_table[result_1_hash_mask_id + tid];
        }

        cal2_pre_call<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_table_high, mem_id,
            result_1_htl, result_2_htl_31, result_2_hash_mask, tid);



    } else {
        size_t i = b + 1;
        __shared__ unsigned char result_2_hash_mask[16];
        __shared__ unsigned char result_4_hash_mask[16];

        for (; i < e - 2; i += 2) {
            Key_T result_0_key = keys[i];

            
            Count_T *result_1_htl;
            size_t result_1_hash_mask_id;
            cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                result_1_hash_mask_id, &result_1_htl,
                hash_table_low, n_low,
                hash,
                seed, s_sz,
                tid, preload_key);

            Count_T result_2_htl_31 = result_1_htl[31];
            if (tid < 16) {
                result_2_hash_mask[tid] = hash_mask_table[result_1_hash_mask_id + tid];
            }

            Count_T *result_3_htl;
            size_t result_3_hash_mask_id;
            cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                result_3_hash_mask_id, &result_3_htl,
                hash_table_low, n_low,
                hash,
                seed, s_sz,
                tid, result_0_key);

            Count_T result_4_htl_31 = result_3_htl[31];
            if (tid < 16) {
                result_4_hash_mask[tid] = hash_mask_table[result_3_hash_mask_id + tid];
            }

            cal2_pre_call<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, mem_id,
                result_1_htl, result_2_htl_31, result_2_hash_mask, tid);

            preload_key = keys[i + 1];

            cal2_pre_call<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, mem_id,
                result_3_htl, result_4_htl_31, result_4_hash_mask, tid);    
        }

        Key_T result_0_key = keys[i];
            
        Count_T *result_1_htl;
        size_t result_1_hash_mask_id;
        cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            result_1_hash_mask_id, &result_1_htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, preload_key);

        Count_T result_2_htl_31 = result_1_htl[31];
        if (tid < 16) {
            result_2_hash_mask[tid] = hash_mask_table[result_1_hash_mask_id + tid];
        }

        Count_T *result_3_htl;
        size_t result_3_hash_mask_id;
        cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            result_3_hash_mask_id, &result_3_htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, result_0_key);

        Count_T result_4_htl_31 = result_3_htl[31];
        if (tid < 16) {
            result_4_hash_mask[tid] = hash_mask_table[result_3_hash_mask_id + tid];
        }

        cal2_pre_call<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_table_high, mem_id,
            result_1_htl, result_2_htl_31, result_2_hash_mask, tid);

        cal2_pre_call<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_table_high, mem_id,
            result_3_htl, result_4_htl_31, result_4_hash_mask, tid);    

    }  
}


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_warp_mem_threadfence_preload_pre_cal(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz,
    unsigned char *hash_mask_table,
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

    Key_T preload_key;
    if (b < sz) {
        preload_key = keys[b];
    }

    if ((e - b) % 2 == 1) {
        size_t i = b + 1;


        __shared__ unsigned char result_2_hash_mask[16];
        __shared__ unsigned char result_4_hash_mask[16];


        for (; i < e; i += 2) {
            Key_T result_0_key = keys[i];
            
            
            Count_T *result_1_htl;
            size_t result_1_hash_mask_id;
            cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                result_1_hash_mask_id, &result_1_htl,
                hash_table_low, n_low,
                hash,
                seed, s_sz,
                tid, preload_key);

            Count_T result_2_cv = result_1_htl[tid]; // load 2
            Count_T result_2_htl_31 = result_1_htl[31];
            if (tid < 16) {
                result_2_hash_mask[tid] = hash_mask_table[result_1_hash_mask_id + tid];
            }
            
            Count_T *result_3_htl;
            size_t result_3_hash_mask_id;
            cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                result_3_hash_mask_id, &result_3_htl,
                hash_table_low, n_low,
                hash,
                seed, s_sz,
                tid, result_0_key);

            Count_T result_4_cv = result_3_htl[tid]; // load 2
            Count_T result_4_htl_31 = result_3_htl[31];
            if (tid < 16) {
                result_4_hash_mask[tid] = hash_mask_table[result_3_hash_mask_id + tid];
            }

            search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, count + i - 1, result_2_cv, result_2_htl_31, result_2_hash_mask, tid);

            preload_key = keys[i + 1];
    
            search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, count + i, result_4_cv, result_4_htl_31, result_4_hash_mask, tid);       

        }

        Count_T *result_1_htl;
        size_t result_1_hash_mask_id;
        cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            result_1_hash_mask_id, &result_1_htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, preload_key);

        Count_T result_2_cv = result_1_htl[tid]; // load 2
        Count_T result_2_htl_31 = result_1_htl[31];
        if (tid < 16) {
            result_2_hash_mask[tid] = hash_mask_table[result_1_hash_mask_id + tid];
        }

        search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, count + e - 1, result_2_cv, result_2_htl_31, result_2_hash_mask, tid);


    } else {
        size_t i = b + 1;
        __shared__ unsigned char result_2_hash_mask[16];
        __shared__ unsigned char result_4_hash_mask[16];

        for (; i < e - 2; i += 2) {
            Key_T result_0_key = keys[i];
            
            Count_T *result_1_htl;
            size_t result_1_hash_mask_id;
            cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                result_1_hash_mask_id, &result_1_htl,
                hash_table_low, n_low,
                hash,
                seed, s_sz,
                tid, preload_key);

            Count_T result_2_cv = result_1_htl[tid]; // load 2
            Count_T result_2_htl_31 = result_1_htl[31];
            if (tid < 16) {
                result_2_hash_mask[tid] = hash_mask_table[result_1_hash_mask_id + tid];
            }

            
            Count_T *result_3_htl;
            size_t result_3_hash_mask_id;
            cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                result_3_hash_mask_id, &result_3_htl,
                hash_table_low, n_low,
                hash,
                seed, s_sz,
                tid, result_0_key);

            Count_T result_4_cv = result_3_htl[tid]; // load 2
            Count_T result_4_htl_31 = result_3_htl[31];
            if (tid < 16) {
                result_4_hash_mask[tid] = hash_mask_table[result_3_hash_mask_id + tid];
            }

            search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, count + i - 1, result_2_cv, result_2_htl_31, result_2_hash_mask, tid);

            preload_key = keys[i + 1];

            search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, count + i, result_4_cv, result_4_htl_31, result_4_hash_mask, tid); 
        }

        Key_T result_0_key = keys[i];

        Count_T *result_1_htl;
        size_t result_1_hash_mask_id;
        cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            result_1_hash_mask_id, &result_1_htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, preload_key);

        Count_T result_2_cv = result_1_htl[tid]; // load 2
        Count_T result_2_htl_31 = result_1_htl[31];
        if (tid < 16) {
            result_2_hash_mask[tid] = hash_mask_table[result_1_hash_mask_id + tid];
        }

        Count_T *result_3_htl;
        size_t result_3_hash_mask_id;
        cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            result_3_hash_mask_id, &result_3_htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, result_0_key);

        Count_T result_4_cv = result_3_htl[tid]; // load 2
        Count_T result_4_htl_31 = result_3_htl[31];
        if (tid < 16) {
            result_4_hash_mask[tid] = hash_mask_table[result_3_hash_mask_id + tid];
        }

        search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_table_high, count + i - 1, result_2_cv, result_2_htl_31, result_2_hash_mask, tid);

        search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_table_high, count + i, result_4_cv, result_4_htl_31, result_4_hash_mask, tid); 

    }  
}
