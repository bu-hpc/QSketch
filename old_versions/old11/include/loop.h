#pragma once

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void cal1_pre_cal_sub_warp_loop(
    size_t *hash_mask_id, Count_T **htl, unsigned int *need_insert,
    
    Count_T *hash_table_low, size_t n_low,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz,
    
    unsigned char tid, unsigned char sub_warp_id, unsigned char sub_warp_tid, unsigned int sub_warp_mask, const Key_T &v,

    size_t j
    ) {
    Hashed_T hv = hash(seed + sub_warp_tid * s_sz, s_sz, v);
    Hashed_T htl_offset;

    if (tid % SUB_WARP_SIZE == 0) {
        htl_offset = (hv % n_low) * SUB_WARP_SIZE;
        *hash_mask_id = (hv % HASH_MASK_TABLE_SIZE_SUB_WARP) * HASH_MASK_SIZE_SUB_WARP;
    }

    __syncwarp(0xffffffff);
    htl_offset = __shfl_sync(sub_warp_mask, htl_offset, sub_warp_id * SUB_WARP_SIZE);
    *htl = hash_table_low + htl_offset;


    *hash_mask_id = __shfl_sync(sub_warp_mask, *hash_mask_id, sub_warp_id * SUB_WARP_SIZE);
}


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void cal2_pre_call_sub_warp_loop(Count_T cv, Count_T *hash_table_high, unsigned int *mem_id,
    Count_T *htl, Count_T htl_7, unsigned char *hash_mask, 
    unsigned char tid, unsigned char sub_warp_id, unsigned char sub_warp_tid, unsigned int sub_warp_mask, Key_T v, bool need_insert_nii)
{
    unsigned int warp_mask_id = tid/8;
    unsigned char warp_mask = 1u << (tid % 8);

    __syncwarp(0xffffffff);
    unsigned int as = __any_sync(0xffffffff, htl_7 > BUFFER_START);
    
    if ( __any_sync(0xffffffff, htl_7 > BUFFER_START)
        // htl_7 > BUFFER_START
        ) {
        #pragma unroll
        for (size_t i = 0; i < (WARP_SIZE); i += SUB_WARP_SIZE)
        {
            Count_T id = __shfl_sync(0xffffffff, htl_7, i);
            __syncwarp(0xffffffff);
            if (id > BUFFER_START) {
                unsigned char *hash_mask_high = hash_mask + (i / SUB_WARP_SIZE) * HASH_MASK_SIZE_SUB_WARP;
                insert_high_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_table_high, id, hash_mask_high, tid, warp_mask_id, warp_mask, v); 
            }
        }
    }

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

    if (add != 0 && (update_low == 0) && need_insert_nii) {
        atomicAdd(htl + sub_warp_tid, add);
    }
    
    if (update_low_warp) {
        unsigned int id = 0;
        if (tid % SUB_WARP_SIZE == 0 && update_low) {
            id = atomicMalloc_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(htl + 7, mem_id);
        }
        __threadfence_block();

        #pragma unroll
        for (unsigned int idf = 0; idf < WARP_SIZE; idf += SUB_WARP_SIZE) {
            unsigned int mask = __activemask();
            unsigned int idv = __shfl_sync(0xffffffff, id, idf);
            
            if (idv > BUFFER_START){
                unsigned char *hash_mask_high = hash_mask + (idf / SUB_WARP_SIZE) * HASH_MASK_SIZE_SUB_WARP;
                insert_high_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_table_high, idv, hash_mask_high, tid, warp_mask_id, warp_mask, v);
            }
        }
    }
} 

#define LOOP_GROUP_NUM 32


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_without_preload_pre_cal_sub_warp_loop(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    unsigned char *hash_mask_table,
    size_t work_load_per_warp,
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
    __shared__ Key_T buffer[WARP_SIZE * 2];
    __shared__ unsigned int buffer_size;
    // unsigned int buffer_b = 0;
    // unsigned int buffer_e = 0;

    size_t gs = ceil<size_t>(n_low, LOOP_GROUP_NUM);
    if (tid == 0) {
        buffer_size = 0;
    }
    __threadfence_block();

    for (size_t j = 1; j <= LOOP_GROUP_NUM; ++j) {
        size_t low_b = (j - 1) * gs;
        size_t low_e = (j * gs);

        size_t i = b;
        while (i < e) {
            __threadfence_block();
            if (buffer_size < 4) {
                // fill
                if (i + tid < e) {
                    Key_T v = keys[i + tid];
                    Hashed_T hv = hash(seed, s_sz, v);
                    Hashed_T htl_offset = hv % n_low;
                    if (htl_offset >= low_b && htl_offset < low_e) {
                        unsigned int old = atomicAdd(&buffer_size, 1);
                        buffer[old] = v;
                    }
                }
                
                i += WARP_SIZE;
            } else {
                unsigned int ni = buffer_size / 4;
                unsigned int r = buffer_size - ni * 4;
                Key_T old_k;
                if (tid < r) {
                    old_k = buffer[ni * 4 + tid];
                }
                for (unsigned int nii = 0; nii < ni; ++nii) {
                    // Key_T v = keys[i + sub_warp_id]; // load 1
                    Key_T v = buffer[nii * 4 + sub_warp_id];

                    Count_T *htl;
                    size_t hash_mask_id;

                    cal1_pre_cal_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                        &hash_mask_id, &htl,
                        hash_table_low, n_low,
                        hash,
                        seed, s_sz,
                        tid, sub_warp_id, sub_warp_tid, sub_warp_mask, v);
                    Count_T htl_7 = htl[7];
                    Count_T cv = htl[sub_warp_tid];
                    if (tid % 2 == 0) {
                        hash_mask[tid / 2] = hash_mask_table[hash_mask_id + sub_warp_tid / 2];
                    }
                    
                    __threadfence_block();
                    cal2_pre_call_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
                        cv, hash_table_high, mem_id,
                        htl, htl_7, hash_mask, tid, sub_warp_id, sub_warp_tid, sub_warp_mask, v);
                }

                __threadfence_block();
                if (tid < r) {
                    buffer[tid] = old_k;
                }
                if (tid == 0) {
                    atomicSub(&buffer_size, 4 * ni);
                }
            }
            
        }

    }



    unsigned int ni = buffer_size / 4;
    // unsigned int r = buffer_size - ni * 4;
    
    // if (tid == 0 && r != 0) {
    //     printf("err: %u\n", r);
    // }

    for (unsigned int nii = 0; nii < ni; ++nii) {
        // Key_T v = keys[i + sub_warp_id]; // load 1
        // Key_T v = buffer[nii * 4 + sub_warp_id];
        Key_T v = 0;
        bool need_insert_nii = false;
        if (nii * 4 + sub_warp_id < buffer_size) {
            v = buffer[nii * 4 + sub_warp_id];
            need_insert_nii = true;
        }

        Count_T *htl = hash_table_low;
        size_t hash_mask_id = 0;

        cal1_pre_cal_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            &hash_mask_id, &htl,
            hash_table_low, n_low,
            hash,
            seed, s_sz,
            tid, sub_warp_id, sub_warp_tid, sub_warp_mask, v);
        Count_T htl_7;
        Count_T cv;
        if (need_insert_nii) {
            htl_7 = htl[7];
            cv = htl[sub_warp_tid];
        }
         
        if (tid % 2 == 0) {
            hash_mask[tid / 2] = hash_mask_table[hash_mask_id + sub_warp_tid / 2];
        }
        
        __threadfence_block();
        cal2_pre_call_sub_warp_loop<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            cv, hash_table_high, mem_id,
            htl, htl_7, hash_mask, tid, sub_warp_id, sub_warp_tid, sub_warp_mask, v, need_insert_nii);
    }
}

#define LOOP_HASH_MASK_SIZE 3

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_thread_mem_threadfence_without_preload_pre_cal_sub_warp_loop(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    unsigned char *hash_mask_table,
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    unsigned char tid = threadIdx.x % WARP_SIZE;

    // unsigned char sub_warp_id = tid / SUB_WARP_SIZE;
    // unsigned char sub_warp_tid = tid % SUB_WARP_SIZE;

    // unsigned int sub_warp_mask = 0xff << (sub_warp_id * 8);

    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }

    unsigned char hash_mask[LOOP_HASH_MASK_SIZE];
    
    // __shared__ unsigned char hash_mask[16];
    // __shared__ Key_T buffer[WARP_SIZE * 2];
    // __shared__ unsigned int buffer_size;
    // unsigned int buffer_b = 0;
    // unsigned int buffer_e = 0;

    size_t gs = ceil<size_t>(n_low, LOOP_GROUP_NUM);
    // if (tid == 0) {
    //     buffer_size = 0;
    // }
    // __threadfence_block();

    // for (size_t j = 1; j <= LOOP_GROUP_NUM; ++j) {
    //     size_t low_b = (j - 1) * gs;
    //     size_t low_e = (j * gs);
    //     for (size_t i = b + tid; i < e; i += WARP_SIZE) {
    //         Key_T v = keys[i];
    //         Hashed_T hv = hash(seed, s_sz, v);
    //         Hashed_T htl_offset = hv % n_low;
    //         if (htl_offset >= low_b && htl_offset < low_e) {
    //             // unsigned int old = atomicAdd(&buffer_size, 1);
    //             // buffer[old] = v;
    //             Count_T *htl = hash_table_low + htl_offset * SUB_WARP_SIZE;
    //             size_t hash_mask_id = (hv % HASH_MASK_SIZE_SUB_WARP) * HASH_MASK_SIZE_SUB_WARP;

    //             if (htl[7] > BUFFER_START) {
    //                 Count_T *hth = hash_table_high + (htl[7] - BUFFER_START) * WARP_SIZE;

    //             } else {

    //             }

    //         }
    //     }
    // }


    // for (size_t j = 1; j <= LOOP_GROUP_NUM; ++j) {
    //     size_t low_b = (j - 1) * gs;
    //     size_t low_e = (j * gs);

    //     size_t i = b;
    //     while (i < e) {
    //         __threadfence_block();
    //         if (buffer_size < 4) {
    //             // fill
    //             if (i + tid < e) {
    //                 Key_T v = keys[i + tid];
    //                 Hashed_T hv = hash(seed, s_sz, v);
    //                 Hashed_T htl_offset = hv % n_low;
    //                 if (htl_offset >= low_b && htl_offset < low_e) {
    //                     unsigned int old = atomicAdd(&buffer_size, 1);
    //                     buffer[old] = v;
    //                 }
    //             }
                
    //             i += WARP_SIZE;
    //         } else {
    //             unsigned int ni = buffer_size / 4;
    //             unsigned int r = buffer_size - ni * 4;
    //             Key_T old_k;
    //             if (tid < r) {
    //                 old_k = buffer[ni * 4 + tid];
    //             }
    //             for (unsigned int nii = 0; nii < ni; ++nii) {
    //                 // Key_T v = keys[i + sub_warp_id]; // load 1
    //                 Key_T v = buffer[nii * 4 + sub_warp_id];

    //                 Count_T *htl;
    //                 size_t hash_mask_id;

    //                 cal1_pre_cal_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
    //                     &hash_mask_id, &htl,
    //                     hash_table_low, n_low,
    //                     hash,
    //                     seed, s_sz,
    //                     tid, sub_warp_id, sub_warp_tid, sub_warp_mask, v);
    //                 Count_T htl_7 = htl[7];
    //                 Count_T cv = htl[sub_warp_tid];
    //                 if (tid % 2 == 0) {
    //                     hash_mask[tid / 2] = hash_mask_table[hash_mask_id + sub_warp_tid / 2];
    //                 }
                    
    //                 __threadfence_block();
    //                 cal2_pre_call_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
    //                     cv, hash_table_high, mem_id,
    //                     htl, htl_7, hash_mask, tid, sub_warp_id, sub_warp_tid, sub_warp_mask, v);
    //             }

    //             __threadfence_block();
    //             if (tid < r) {
    //                 buffer[tid] = old_k;
    //             }
    //             if (tid == 0) {
    //                 atomicSub(&buffer_size, 4 * ni);
    //             }
    //         }
            
    //     }

    // }



    // unsigned int ni = buffer_size / 4;
    // // unsigned int r = buffer_size - ni * 4;
    
    // // if (tid == 0 && r != 0) {
    // //     printf("err: %u\n", r);
    // // }

    // for (unsigned int nii = 0; nii < ni; ++nii) {
    //     // Key_T v = keys[i + sub_warp_id]; // load 1
    //     // Key_T v = buffer[nii * 4 + sub_warp_id];
    //     Key_T v = 0;
    //     bool need_insert_nii = false;
    //     if (nii * 4 + sub_warp_id < buffer_size) {
    //         v = buffer[nii * 4 + sub_warp_id];
    //         need_insert_nii = true;
    //     }

    //     Count_T *htl = hash_table_low;
    //     size_t hash_mask_id = 0;

    //     cal1_pre_cal_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
    //         &hash_mask_id, &htl,
    //         hash_table_low, n_low,
    //         hash,
    //         seed, s_sz,
    //         tid, sub_warp_id, sub_warp_tid, sub_warp_mask, v);
    //     Count_T htl_7;
    //     Count_T cv;
    //     if (need_insert_nii) {
    //         htl_7 = htl[7];
    //         cv = htl[sub_warp_tid];
    //     }
         
    //     if (tid % 2 == 0) {
    //         hash_mask[tid / 2] = hash_mask_table[hash_mask_id + sub_warp_tid / 2];
    //     }
        
    //     __threadfence_block();
    //     cal2_pre_call_sub_warp_loop<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
    //         cv, hash_table_high, mem_id,
    //         htl, htl_7, hash_mask, tid, sub_warp_id, sub_warp_tid, sub_warp_mask, v, need_insert_nii);
    // }
}