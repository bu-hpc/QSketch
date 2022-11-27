#pragma once


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void cal2_pre_call_sub_warp_loop_se(Count_T cv, Count_T *hash_table_high, unsigned int *mem_id,
    Count_T *htl, Count_T htl_7, unsigned char *hash_mask, 
    unsigned char tid, unsigned char sub_warp_id, unsigned char sub_warp_tid, unsigned int sub_warp_mask, 
    Key_T v,
    Count_T *hash_table_low_s, Count_T *hash_table_low_e)
{
    unsigned int warp_mask_id = tid/8;
    unsigned char warp_mask = 1u << (tid % 8);

    // __syncwarp(0xffffffff);
    // unsigned int as = __any_sync(0xffffffff, htl_7 > BUFFER_START);
    
    // if ( __any_sync(0xffffffff, htl_7 > BUFFER_START)
    //     // htl_7 > BUFFER_START
    //     ) {
    //     #pragma unroll
    //     for (size_t i = 0; i < (WARP_SIZE); i += SUB_WARP_SIZE)
    //     {
    //         Count_T id = __shfl_sync(0xffffffff, htl_7, i);
    //         __syncwarp(0xffffffff);
    //         if (id > BUFFER_START) {
    //             unsigned char *hash_mask_high = hash_mask + (i / SUB_WARP_SIZE) * HASH_MASK_SIZE_SUB_WARP;
    //             insert_high_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_table_high, id, hash_mask_high, tid, warp_mask_id, warp_mask, v); 
    //         }
    //     }
    // }

    __syncwarp(0xffffffff);
    unsigned int insert_low = __any_sync(sub_warp_mask, htl_7 > BUFFER_START);

    Count_T max_count = 0;
    Count_T add = 0;

    if (insert_low == 0){
        unsigned char thm = hash_mask[tid/2];
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
    }
    __syncwarp(0xffffffff);
    unsigned int update_low = __any_sync(sub_warp_mask, max_count > 128);
    unsigned int update_low_warp = __any_sync(0xffffffff, max_count > 128);

    if (add != 0 && (update_low == 0)) {
        Count_T *ptr = htl + sub_warp_tid;
        if (ptr >= hash_table_low_s && ptr < hash_table_low_e) 
        {
            atomicAdd(ptr, add);
        }
        
    }
    
    // if (update_low_warp) {
    //     unsigned int id = 0;
    //     if (tid % SUB_WARP_SIZE == 0 && update_low) {
    //         id = atomicMalloc_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(htl + 7, mem_id);
    //     }
    //     __threadfence_block();

    //     #pragma unroll
    //     for (unsigned int idf = 0; idf < WARP_SIZE; idf += SUB_WARP_SIZE) {
    //         unsigned int mask = __activemask();
    //         unsigned int idv = __shfl_sync(0xffffffff, id, idf);
            
    //         if (idv > BUFFER_START){
    //             unsigned char *hash_mask_high = hash_mask + (idf / SUB_WARP_SIZE) * HASH_MASK_SIZE_SUB_WARP;
    //             insert_high_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_table_high, idv, hash_mask_high, tid, warp_mask_id, warp_mask, v);
    //         }
    //     }
    // }
} 


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_without_preload_pre_cal_sub_warp_loop2(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    unsigned char *hash_mask_table,
    size_t work_load_per_warp,
    // size_t j,
    Count_T *hash_table_low_s, Count_T *hash_table_low_e,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    unsigned char tid = threadIdx.x % WARP_SIZE;
    unsigned char sub_warp_id = tid / SUB_WARP_SIZE;
    unsigned char sub_warp_tid = tid % SUB_WARP_SIZE;

    unsigned int sub_warp_mask = 0xff << (sub_warp_id * 8);

    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }
    
    __shared__ unsigned char hash_mask[16];

    for (size_t i = b; i < e; i += 4) {
        
        Key_T v = keys[i + sub_warp_id]; // load 1

        Count_T *htl;
        size_t hash_mask_id;

        cal1_pre_cal_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            &hash_mask_id, &htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, sub_warp_id, sub_warp_tid, sub_warp_mask, v
            );

        Count_T htl_7 = 0;
        Count_T cv = 0;
        if (htl >= hash_table_low_s && htl < hash_table_low_e) {
            htl_7 = htl[7];
            cv = htl[sub_warp_tid];
            if (tid % 2 == 0) {
                hash_mask[tid / 2] = hash_mask_table[hash_mask_id + sub_warp_tid / 2];
            }
        }

        __threadfence_block();

        cal2_pre_call_sub_warp_loop_se<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            cv, hash_table_high, mem_id,
            htl, htl_7, hash_mask, tid, sub_warp_id, sub_warp_tid, sub_warp_mask, v, 
            hash_table_low_s, hash_table_low_e);
    }
}
