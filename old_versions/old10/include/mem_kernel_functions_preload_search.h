#pragma once

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline Count_T search_low(unsigned char *hash_mask, Count_T cv, size_t tid) {

    Count_T thread_min_low = 0xffffffff;
    unsigned char thm = (hash_mask[tid/2]);
    thm = thm >> ((tid % 2) * 4);
    if (thm & 0b00000001) {
        thread_min_low = min(thread_min_low, (cv & 0x000000ffu));
    }

    if (thm & 0b00000010) {
        thread_min_low = min(thread_min_low, (cv & 0x0000ff00u) >> 8);
    }

    if (thm & 0b00000100) {
        thread_min_low = min(thread_min_low, (cv & 0x00ff0000u) >> 16);
    }

    if (thm & 0b00001000) {
        thread_min_low = min(thread_min_low, (cv & 0xff000000u) >> 24);
    }

    return thread_min_low;
}
template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline Count_T search_high(unsigned char *hash_mask, Count_T *hth, size_t tid) {

    Count_T thread_min_high = 0xffffffff;
    unsigned int warp_mask_id = tid/8;
    unsigned char warp_mask = 1u << (tid % 8);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (hash_mask[j * 4 + warp_mask_id] & warp_mask) {
            thread_min_high = min(thread_min_high, hth[j * WARP_SIZE + tid]);
        }
    }
    return thread_min_high;
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void search_calculation_2(Count_T *hash_table_high, Count_T *count, Count_T cv, Count_T htl_31,
    unsigned char *hash_mask, size_t tid)
{
    // Count_T htl_31 = __shfl_sync(0xffffffff, cv, 31);
    Count_T thread_min_low = search_low<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_mask, cv, tid);

    if (htl_31 > BUFFER_START) {
        Count_T *hth = hash_table_high + (htl_31 - BUFFER_START) * 128;
        Count_T thread_min_high = search_high<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_mask, hth, tid);
        // count_min = calculate_min_two_levels
        for (int j = 16; j >= 1; j = j >> 1) {
            Count_T t_low = __shfl_down_sync(0xffffffff, thread_min_low, j);
            Count_T t_high = __shfl_down_sync(0xffffffff, thread_min_high, j);
            if (tid < j) {
                thread_min_low = min(thread_min_low, t_low);
                thread_min_high = min(thread_min_high, t_high);
            }
        }
        if (tid == 0) {
            *count = thread_min_low + thread_min_high;
        }
    } else {
        // count_min = calculate_min
        for (int j = 16; j >= 1; j = j >> 1) {
            Count_T t_low = __shfl_down_sync(0xffffffff, thread_min_low, j);
            if (tid < j) {
                thread_min_low = min(thread_min_low, t_low);
            }
        }
        if (tid == 0) {
            *count = thread_min_low;
        }
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_warp_mem_threadfence_without_preload(Key_T *keys, size_t sz,
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
    
    __shared__ unsigned char hash_mask[16];

    for (size_t i = b; i < e; ++i) {
        Key_T v = keys[i]; // load 1
        Count_T *htl;

        // search_calculation_1
        cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_mask, &htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, v);

        // Count_T htl_31 = htl[31];
        Count_T cv = htl[tid]; // load 2
        Count_T htl_31 = htl[31];

        // search calculation 2
        search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_table_high, count + i, cv, htl_31, hash_mask, tid);
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_warp_mem_threadfence_preload(Key_T *keys, size_t sz,
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

    Key_T preload_key;
    if (b < sz) {
        preload_key = keys[b];
    }

    if ((e - b) % 2 == 1) {
        size_t i = b + 1;
        for (; i < e; i += 2) {
            Key_T result_0_key = keys[i];
            
            __shared__ unsigned char result_1_hash_mask[16];
            Count_T *result_1_htl;
            cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                result_1_hash_mask, &result_1_htl,
                hash_table_low, n_low,
                hash,
                seed, s_sz,
                tid, preload_key);

            Count_T result_2_cv = result_1_htl[tid]; // load 2
            Count_T result_2_htl_31 = result_1_htl[31];

            __shared__ unsigned char result_3_hash_mask[16];
            Count_T *result_3_htl;
            cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                result_3_hash_mask, &result_3_htl,
                hash_table_low, n_low,
                hash,
                seed, s_sz,
                tid, result_0_key);

            Count_T result_4_cv = result_3_htl[tid]; // load 2
            Count_T result_4_htl_31 = result_3_htl[31];

            // cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            //     hash_table_high, mem_id,
            //     result_1_htl, result_2_htl_31, result_1_hash_mask, tid);
            search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, count + i - 1, result_2_cv, result_2_htl_31, result_1_hash_mask, tid);

            preload_key = keys[i + 1];

            // cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            //     hash_table_high, mem_id,
            //     result_3_htl, result_4_htl_31, result_3_hash_mask, tid);    
            search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, count + i, result_4_cv, result_4_htl_31, result_3_hash_mask, tid);       

        }

        __shared__ unsigned char result_1_hash_mask[16];
        Count_T *result_1_htl;
        cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            result_1_hash_mask, &result_1_htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, preload_key);

        Count_T result_2_cv = result_1_htl[tid]; // load 2
        Count_T result_2_htl_31 = result_1_htl[31];

        // cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
        //     hash_table_high, mem_id,
        //     result_1_htl, result_2_htl_31, result_1_hash_mask, tid);
        search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, count + e - 1, result_2_cv, result_2_htl_31, result_1_hash_mask, tid);


    } else {
        size_t i = b + 1;
        for (; i < e - 2; i += 2) {
            Key_T result_0_key = keys[i];
            
            __shared__ unsigned char result_1_hash_mask[16];
            Count_T *result_1_htl;
            cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                result_1_hash_mask, &result_1_htl,
                hash_table_low, n_low,
                hash,
                seed, s_sz,
                tid, preload_key);

            Count_T result_2_cv = result_1_htl[tid]; // load 2
            Count_T result_2_htl_31 = result_1_htl[31];

            __shared__ unsigned char result_3_hash_mask[16];
            Count_T *result_3_htl;
            cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                result_3_hash_mask, &result_3_htl,
                hash_table_low, n_low,
                hash,
                seed, s_sz,
                tid, result_0_key);

            Count_T result_4_cv = result_3_htl[tid]; // load 2
            Count_T result_4_htl_31 = result_3_htl[31];

            // cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            //     hash_table_high, mem_id,
            //     result_1_htl, result_2_htl_31, result_1_hash_mask, tid);
            search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, count + i - 1, result_2_cv, result_2_htl_31, result_1_hash_mask, tid);

            preload_key = keys[i + 1];

            // cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            //     hash_table_high, mem_id,
            //     result_3_htl, result_4_htl_31, result_3_hash_mask, tid);    
            search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                hash_table_high, count + i, result_4_cv, result_4_htl_31, result_3_hash_mask, tid); 
        }

        Key_T result_0_key = keys[i];
            
        __shared__ unsigned char result_1_hash_mask[16];
        Count_T *result_1_htl;
        cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            result_1_hash_mask, &result_1_htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, preload_key);

        Count_T result_2_cv = result_1_htl[tid]; // load 2
        Count_T result_2_htl_31 = result_1_htl[31];

        __shared__ unsigned char result_3_hash_mask[16];
        Count_T *result_3_htl;
        cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            result_3_hash_mask, &result_3_htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, result_0_key);

        Count_T result_4_cv = result_3_htl[tid]; // load 2
        Count_T result_4_htl_31 = result_3_htl[31];

        // cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
        //     hash_table_high, mem_id,
        //     result_1_htl, result_2_htl_31, result_1_hash_mask, tid);
        search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_table_high, count + i - 1, result_2_cv, result_2_htl_31, result_1_hash_mask, tid);

        // cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
        //     hash_table_high, mem_id,
        //     result_3_htl, result_4_htl_31, result_3_hash_mask, tid);    
        search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_table_high, count + i, result_4_cv, result_4_htl_31, result_3_hash_mask, tid); 

        // Key_T result_0_key = keys[i];
            
        // __shared__ unsigned char result_1_hash_mask[16];
        // Count_T *result_1_htl;
        // cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
        //     result_1_hash_mask, &result_1_htl,
        //     hash_table_low, n_low,
        //     hash,
        //     seed, s_sz,
        //     tid, preload_key);

        // Count_T result_2_htl_31 = result_1_htl[31];

        // __shared__ unsigned char result_3_hash_mask[16];
        // Count_T *result_3_htl;
        // cal1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
        //     result_3_hash_mask, &result_3_htl,
        //     hash_table_low, n_low,
        //     hash,
        //     seed, s_sz,
        //     tid, result_0_key);

        // Count_T result_4_htl_31 = result_3_htl[31];

        // cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
        //     hash_table_high, mem_id,
        //     result_1_htl, result_2_htl_31, result_1_hash_mask, tid);

        // cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
        //     hash_table_high, mem_id,
        //     result_3_htl, result_4_htl_31, result_3_hash_mask, tid);    

    }  
}
