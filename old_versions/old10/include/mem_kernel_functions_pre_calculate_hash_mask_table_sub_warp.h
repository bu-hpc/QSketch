#pragma once

#define SUB_WARP_SIZE 8

// template <typename Count_T>
// __global__ void check_count(Count_T *count, size_t sz) {
//     for (size_t i = threadIdx.x; i < sz; i += 32) {
//         atomicMin(count + i, 100);
//     }
// }

// cal1_pre_cal fast
template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void cal1_pre_cal_sub_warp(
    size_t *hash_mask_id, Count_T **htl,
    
    Count_T *hash_table_low, size_t n_low,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz,
    
    unsigned char tid, unsigned char sub_warp_id, unsigned char sub_warp_tid, unsigned int sub_warp_mask, const Key_T &v
    ) {
    Hashed_T hv = hash(seed + sub_warp_tid * s_sz, s_sz, v);
    Hashed_T htl_offset;

    if (tid % SUB_WARP_SIZE == 0) {
        htl_offset = (hv % n_low) * SUB_WARP_SIZE;
        *hash_mask_id = (hv % HASH_MASK_TABLE_SIZE_SUB_WARP) * HASH_MASK_SIZE_SUB_WARP;
    }

    // __syncwarp(0xffffffff);
    htl_offset = __shfl_sync(sub_warp_mask, htl_offset, sub_warp_id * SUB_WARP_SIZE);
    *htl = hash_table_low + htl_offset;
    *hash_mask_id = __shfl_sync(sub_warp_mask, *hash_mask_id, sub_warp_id * SUB_WARP_SIZE);


    // __threadfence_block();
    // if (v == 3439647702u) {
    //     printf("cal1_pre_cal_sub_warp tid: %d, htl_offset: %u, hash_mask_id: %lu\n", int(tid), htl_offset, *hash_mask_id);
    // }

}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void insert_high_sub_warp(Count_T *hash_table_high, const Count_T &id, 
    unsigned char *hash_mask, size_t tid,
    const unsigned int &warp_mask_id, const unsigned char &warp_mask, Key_T v) {
    Count_T *hth = hash_table_high + (id - BUFFER_START) * 32;
    
    // if (__any_sync(0xffffffff, v == 3439647702u)) {
    //     printf("before tid: %lu, hthv: %u\n", tid, hth[tid]);
    //     // if (tid < 16) {
    //     //     printf("tid: %lu, hash_mask: %\n", );
    //     // }
    //     // if (hash_mask[warp_mask_id] & warp_mask) {
    //     //     printf("active tid: %lu\n", tid);
    //     // }
    // }
    // __threadfence_block();
    // __syncwarp(0xffffffff);
    if (hash_mask[warp_mask_id] & warp_mask) {
        atomicAdd(hth + tid, 1);
    }

    // if (__any_sync(0xffffffff, v == 81758573u)) {
    //     printf("after tid: %lu, hthv: %u\n", tid, hth[tid]);
    // }

    // if (v == 81758573u) {
    //     if (tid % SUB_WARP_SIZE == 0){
    //         printf("insert_high address: %lu\n", hth);
    //     }
    // }

    // __threadfence_block();

    // __any_sync(0xffffffff, v == 81758573u);
    // __all_sync(0xffffffff, v);
    // __syncthreads();
    // __syncwarp(0xffffffff);
    // if (__any_sync(0xffffffff, (v == 3439647702u))) {
    //     // printf("after tid: %lu\n", tid);
    //     // printf("after tid: %lu, hthv: %u\n", tid, hth[tid]);

    //     // if (tid == 0) {
    //         // printf("insert_high\n");
    //         // printf("hash_mask address: %lu\n", hash_mask);
    //         // for (int i = 0; i < 16; ++i) {
    //         //     unsigned int t = hash_mask[i];
    //         //     printf("%u", t);
    //         // }
    //         // printf("\n");

    //     // printf("tid: %lu, hth address: %lu\n", tid, hth);

    //     // if (hash_mask[warp_mask_id] & warp_mask) {
    //     //     // atomicAdd(hth + tid, 1);
    //     //     printf("tid: %lu, hth address: %lu\n", tid, hth);
    //     // }

    //     // }
    // }

    // unsigned int mask = __activemask();
    // if (tid == 0) {
    //     printf("mask :%x\n", mask);
    // }

}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline unsigned int atomicMalloc_sub_warp(Count_T *htl, unsigned int *mem_id) {
    unsigned int id = 0;
    unsigned int old = atomicCAS(htl, 0, 1);
    if (old == 0) {
        id = atomicAdd(mem_id, 1);
        // htl[31] = id;
        *htl = id;
    } else {
        // while (__ldcv(htl + 31) <= 1) {}
        // tem = __ldcv(htl + 31);
        // id = __ldcv(htl + 31);
        while (id <= 1) {
            id = max(id, __ldcv(htl));
            __threadfence_block();
            // printf("wait\n");
        }
    }
    return id;
}



template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void cal2_pre_call_sub_warp(Count_T cv, Count_T *hash_table_high, unsigned int *mem_id,
    Count_T *htl, Count_T htl_7, unsigned char *hash_mask, unsigned char tid, unsigned char sub_warp_id, unsigned char sub_warp_tid, unsigned int sub_warp_mask, Key_T v)
{
    // unsigned char sub_warp_tid = tid % SUB_WARP_SIZE;
    unsigned int warp_mask_id = tid/8;
    unsigned char warp_mask = 1u << (tid % 8);

    // unsigned int mask = __activemask();
    // if (tid == 0) {
    //     // if (mask != 0xffffffff)
    //         // printf("mask :%x\n", mask);
    //         printf("cal2 p1\n");
    // }
    // __threadfence_block();
    // __syncwarp(0xffffffff);
    unsigned int as = __any_sync(0xffffffff, htl_7 > BUFFER_START);
    // if (tid == 0 || tid == 24) {
    //     printf("tid: %d, cal2 p0, as: %u\n", int(tid), as);
    // }
    // printf("tid: %d, htl_7: %u, as: %u\n", int(tid), htl_7, as);

    if ( __any_sync(0xffffffff, htl_7 > BUFFER_START)
        // htl_7 > BUFFER_START
        ) {

        // if (tid == 0 || tid == 24) {
        //     printf("tid: %d, cal2 hp0\n", int(tid));
        // }
        // insert_high_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_table_high, htl_7, hash_mask, tid, warp_mask_id, warp_mask);
        #pragma unroll
        for (size_t i = 0; i < (WARP_SIZE); i += SUB_WARP_SIZE)
        {
            Count_T id = __shfl_sync(0xffffffff, htl_7, i);
            // if (tid == i) {
            //     printf("p1: tid: %d, v: %u\n", int(tid), v);
            // }
            // Key_T ptv = __shfl_sync(0xffffffff, v, i);
            // if (tid == 0) {
            //     printf("p2: tid: %d, ptv: %u\n", int(tid), ptv);
            // }

            // __syncwarp(0xffffffff);

            // if (tid == i) {
            //     printf("p2: tid: %d, v: %u\n", int(tid), v);
            // }

            if (id > BUFFER_START) {

                unsigned char *hash_mask_high = hash_mask + (i / SUB_WARP_SIZE) * HASH_MASK_SIZE_SUB_WARP;
                insert_high_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_table_high, id, hash_mask_high, tid, warp_mask_id, warp_mask, v);
                
                // if (ptv == 3439647702u) {
                //     if (tid == 0)
                //         printf("insert_high case 1, id: %u\n", id);
                // }
                // if (tid == i && v == 3439647702u) {
                //     printf("insert_high case 1, id: %u\n", id);
                //     if (hash_mask[warp_mask_id] & warp_mask) {
                //         atomicAdd(hth + tid, 1);
                //     }
                // }

                if (__any_sync(0xffffffff, (tid == i) && (v == 3439647702u))) {
                    // if (tid == 0) {
                    //     printf("insert_high case 1, id: %u\n", id);
                    // }
                    // Count_T *hth = hash_table_high + (id - BUFFER_START) * 32;
                    // if (hash_mask_high[warp_mask_id] & warp_mask) {
                    //     // atomicAdd(hth + tid, 1);
                    //     printf("tid: %d, hth val: %u\n", int(tid), hth[tid]);
                    // }
                }
            }


        }
        // printf("err htl_7\n");

        // if (tid == 0 || tid == 24) {
        //     printf("tid: %d, cal2 hp2\n", int(tid));
        // }
        
    }

    // if (tid == 0 || tid == 24) {
    //     printf("tid: %d, cal2 p1\n", int(tid));
    // }

    // __syncwarp(0xffffffff);
    unsigned int insert_low = __any_sync(sub_warp_mask, htl_7 > BUFFER_START);
    // printf("tid: %d, htl_7: %u, insert_low: %u\n", int(tid), htl_7, insert_low);

    Count_T max_count = 0;
    Count_T add = 0;

    if (insert_low == 0){

        
        // Count_T cv = htl[sub_warp_id][sub_warp_tid];
        // unsigned char thm = (hash_mask[sub_warp_id][sub_warp_tid/2]);
        unsigned char thm = hash_mask[tid/2];
        // unsigned char thm = hash_mask[0];
        // unsigned char thm = 0xff;
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

        // if (add != 0 ) {
        //     atomicAdd(htl + sub_warp_tid, add);
        // }

        // if (tid == 0 || tid == 24) {
        //     printf("tid: %d, cal2 lp1\n", int(tid));
        // }
    }
    // __syncwarp(0xffffffff);
    unsigned int update_low = __any_sync(sub_warp_mask, max_count > 128);
    unsigned int update_low_warp = __any_sync(0xffffffff, max_count > 128);

    if (add != 0 && (update_low == 0)) {
        atomicAdd(htl + sub_warp_tid, add);
    }
    
    if (update_low_warp) {
        unsigned int id = 0;
        if (tid % SUB_WARP_SIZE == 0 && update_low) {
            // printf("atomicMalloc_sub_warp tid: %d\n", int(tid));
            id = atomicMalloc_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(htl + 7, mem_id);
            // if ((id - BUFFER_START) * 32 > 1024 * 1024) {
            //     printf("err: %u\n", id);
            // }
            // printf("atomicMalloc_sub_warp finished\n");
        }
        __threadfence_block();

        // if (tid == 0 || tid == 24) {
        //     printf("tid: %d, cal2 lp2\n", int(tid));
        // }
        #pragma unroll
        for (unsigned int idf = 0; idf < WARP_SIZE; idf += SUB_WARP_SIZE) {
            unsigned int mask = __activemask();
            // if (tid == 0) {
            //     printf("cal2: idf: %u, %x\n", idf, mask);
            // }
            // if (tid == 24) {
            //     printf("cal2: idf: %u, %x\n", idf, mask);
            // }
            unsigned int idv = __shfl_sync(0xffffffff, id, idf);
            // if (tid == 0) {
            //     printf("cal2: idf: %u, idv: %u\n", idf, idv);
            // }
            if (idv > BUFFER_START){
                // insert_high<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_table_high, idv, hash_mask, tid, warp_mask_id, warp_mask);
                unsigned char *hash_mask_high = hash_mask + (idf / SUB_WARP_SIZE) * HASH_MASK_SIZE_SUB_WARP;
                insert_high_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_table_high, idv, hash_mask_high, tid, warp_mask_id, warp_mask, v);
                // if (v == 3439647702u) {
                //     if (tid % SUB_WARP_SIZE == 0)
                //         printf("insert_high case 2, id: %u\n", id);
                // }
            }
        }
        // if (tid == 0 || tid == 24) {
        //     printf("tid: %d, cal2 lp3\n", int(tid));
        // }
            // insert_high<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_table_high, id, hash_mask, tid, warp_mask_id, warp_mask);
        // if (v == 433799499u) {
        //     if (tid % SUB_WARP_SIZE == 0)
        //         printf("insert_high case 1\n");
        // }
    }

    // if (tid == 0) {
    //     printf("cal2 pf\n");
    // }
} 


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_without_preload_pre_cal_sub_warp(Key_T *keys, size_t sz,
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
    
    // size_t *debug_hash_mask_id = (size_t *)debug;
    __shared__ unsigned char hash_mask[16];

    for (size_t i = b; i < e; i += 4) {

        // unsigned int mask = __activemask();
        // if (tid == 0) {
        //     // if (mask != 0xffffffff)
        //         // printf("mask :%x\n", mask);
        //     printf("loop: i: %lu\n", i);
        // }

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
        // __threadfence_block();
        Count_T htl_7 = htl[7];
        Count_T cv = htl[sub_warp_tid];
        if (tid % 2 == 0) {
            // hash_mask[tid] = hash_mask_table[hash_mask_id[sub_warp_id] + tid % 4];
            hash_mask[tid / 2] = hash_mask_table[hash_mask_id + sub_warp_tid / 2];
            // if (hash_mask_id + sub_warp_tid / 2 >= 1024 * 4) {
            //     printf("err: %lu\n", hash_mask_id + sub_warp_tid / 2);
            // }
        }
        // if (tid % SUB_WARP_SIZE == 0) {
        //     debug_hash_mask_id[i + sub_warp_id] = hash_mask_id;
        // }

        

        __threadfence_block();

        // if (v == 4087563711) {
        //     if (tid % SUB_WARP_SIZE == 0)
        //         printf("hash_mask_id: %lu\n", hash_mask_id);
        // }

        // if (tid == 0) {
        //     // if (mask != 0xffffffff)
        //         // printf("mask :%x\n", mask);
        //     printf("loop: i: %lu p2\n", i);
        // }

        cal2_pre_call_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            cv, hash_table_high, mem_id,
            htl, htl_7, hash_mask, tid, sub_warp_id, sub_warp_tid, sub_warp_mask, v);


        // if (tid == 0) {
        //     // if (mask != 0xffffffff)
        //         // printf("mask :%x\n", mask);
        //     printf("loop: i: %lu p3\n", i);
        // }
        // cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
        //     hash_table_high, mem_id,
        //     htl, htl_31, hash_mask, tid);
    }    
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_preload_pre_cal_sub_warp(Key_T *keys, size_t sz,
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
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline Count_T search_low_sub_warp(unsigned char *hash_mask, Count_T cv, size_t tid) {

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
__device__ inline Count_T search_high_sub_warp(unsigned char *hash_mask, Count_T *hth, size_t tid) {

    Count_T thread_min_high = 0xffffffff;
    unsigned int warp_mask_id = tid/8;
    unsigned char warp_mask = 1u << (tid % 8);

    if (hash_mask[warp_mask_id] & warp_mask) {
        thread_min_high = hth[tid];
    }

    return thread_min_high;
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void search_calculation_2_sub_warp(Count_T *hash_table_high, Count_T *count, Count_T cv, Count_T htl_7,
    unsigned char *hash_mask, size_t tid, unsigned char sub_warp_id, unsigned char sub_warp_tid, unsigned int sub_warp_mask, Key_T v)
{

    // if (__any_sync(0xffffffff, v == 19397573u)) {
    //     if (tid < 16){
    //         printf("tid: %lu, hash_mask: %d\n", tid, int(hash_mask[tid]));
    //     }
    // }

    Count_T thread_min_low = search_low_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_mask, cv, tid);

    __shared__ Count_T thread_min_high[4];

    if (tid < 4) {
        thread_min_high[tid] = 0;
    }
    __threadfence_block();
    if (__any_sync(0xffffffff, htl_7 > BUFFER_START)) {
        #pragma unroll
        for (size_t i = 0; i < (WARP_SIZE); i += SUB_WARP_SIZE)
        {
           
            Count_T id = __shfl_sync(0xffffffff, htl_7, i);
            if (id > BUFFER_START) {
                unsigned char *hash_mask_high = hash_mask + (i / SUB_WARP_SIZE) * HASH_MASK_SIZE_SUB_WARP;
                Count_T thread_min_high_local_sub_warp = 
                    search_high_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>
                        (hash_mask_high, hash_table_high + (id - BUFFER_START) * WARP_SIZE, tid);

                // if (__any_sync(0xffffffff, v == 3439647702u)) {
                //     Count_T *hth = hash_table_high + (id - BUFFER_START) * WARP_SIZE;
                //     // printf("id: %u, tid: %lu, address: %lu, val: %u\n", id, tid, hth, thread_min_high_local_sub_warp);
                    
                //     printf("id: %u, tid: %lu, address: %lu, val: %u, tmhlsw: %u\n", id, tid, hth, hth[tid], thread_min_high_local_sub_warp);
                // }
                for (int j = 16; j >= 1; j = j >> 1) {
                    Count_T t_low = __shfl_down_sync(0xffffffff, thread_min_high_local_sub_warp, j);
                    if (tid < j) {
                        thread_min_high_local_sub_warp = min(thread_min_high_local_sub_warp, t_low);
                    }
                }
                if (tid == 0) {
                    thread_min_high[i / SUB_WARP_SIZE] = thread_min_high_local_sub_warp;
                }

                // if (tid == 0) {
                //     printf("search high\n");
                // }
            }
        }
    }
    __threadfence_block();
    for (int j = 4; j >= 1; j = j >> 1) {
        Count_T t_low = __shfl_down_sync(0xffffffff, thread_min_low, j);
        if ((tid >= sub_warp_id * SUB_WARP_SIZE) && (tid < sub_warp_id * SUB_WARP_SIZE + j)) {
            thread_min_low = min(thread_min_low, t_low);
        }
    }
    // __threadfence_block();
    if (tid % SUB_WARP_SIZE == 0) {
        *count = thread_min_low + thread_min_high[sub_warp_id];

        // if (v == 3439647702u) {
        //     printf("thread_min_low: %u, thread_min_high: %u, htl_7: %u\n", thread_min_low, thread_min_high[sub_warp_id], htl_7);
        //     // printf("htl_7: %u\n", htl_7);

        // }
    }
}



template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_warp_mem_threadfence_without_preload_pre_cal_sub_warp(Key_T *keys, size_t sz,
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
    // size_t *debug_hash_mask_id = (size_t *)debug;

    for (size_t i = b; i < e; i += 4) {

        // if (i + sub_warp_id >= e) {
        //     printf(" err i + sub_warp_id\n");
        // }
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
        // __threadfence_block();
        Count_T htl_7 = htl[7];
        Count_T cv = htl[sub_warp_tid];
        if (tid % 2 == 0) {
            // hash_mask[tid] = hash_mask_table[hash_mask_id[sub_warp_id] + tid % 4];
            hash_mask[tid / 2] = hash_mask_table[hash_mask_id + sub_warp_tid / 2];
            // if (hash_mask_id + sub_warp_tid / 2 >= 1024 * 4) {
            //     printf("err: %lu\n", hash_mask_id + sub_warp_tid / 2);
            // }
        }
        // if (tid % SUB_WARP_SIZE == 0) {
        //     debug_hash_mask_id[i + sub_warp_id] = hash_mask_id;
        // }
        // if (tid % 2 == 0) {
        //     // hash_mask[tid] = hash_mask_table[hash_mask_id[sub_warp_id] + tid % 4];
        //     hash_mask[tid / 2] = hash_mask_table[hash_mask_id + (tid % 8) / 2];
        // }
        __threadfence_block();

        // search_calculation_2_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
        //     cv, hash_table_high, mem_id,
        //     htl, htl_7, hash_mask, tid, sub_warp_id, sub_warp_tid, sub_warp_mask);
        search_calculation_2_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_table_high, count + i + sub_warp_id, cv,
             htl_7, hash_mask, tid, sub_warp_id, sub_warp_tid, sub_warp_mask, v);

        // cal2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
        //     hash_table_high, mem_id,
        //     htl, htl_31, hash_mask, tid);
    }    


    // size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    // size_t tid = threadIdx.x % WARP_SIZE;
    // size_t b = wid * work_load_per_warp;
    // size_t e = (wid + 1) * work_load_per_warp;

    // if (e >= sz) {
    //     e = sz;
    // }
    
    // __shared__ unsigned char hash_mask[16];

    // for (size_t i = b; i < e; ++i) {
    //     Key_T v = keys[i]; // load 1


    //     Count_T *htl;
    //     size_t hash_mask_id;

    //     // search_calculation_1
    //     cal1_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
    //         hash_mask_id, &htl,
    //         hash_table_low, n_low,
    //         hash,
    //         seed, s_sz,
    //         tid, v);

    //     // Count_T htl_31 = htl[31];
    //     Count_T cv = htl[tid]; // load 2
    //     Count_T htl_31 = htl[31];
    //     if (tid < 16) {
    //         hash_mask[tid] = hash_mask_table[hash_mask_id + tid];
    //         // hash_mask[tid] = hash_mask_table[(i % 1024) * 16 + tid];
    //         // hash_mask[tid] = hash_mask_table[(i) * 16 + tid];
    //         // hash_mask[tid] = 0b00001000;
    //     }

    //     // search calculation 2
    //     search_calculation_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
    //         hash_table_high, count + i, cv, htl_31, hash_mask, tid);
    // }
}

