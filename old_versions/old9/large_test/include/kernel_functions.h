#pragma once

/*

function names :
[insert | search] _ [thread | warp ...] _ [...]

thread : each thread should insert one element.
warp : each warp should insert one element.

*/

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_thread(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
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
        for (size_t j = 0; j < n; ++j) {
            Count_T *h = hash_table + j * n;
            Hashed_T hv = hash(seed + j * s_sz, s_sz, v) % n;
            atomicAdd(h + hv, 1);
        }
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_thread(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
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

    Count_T thread_min = Count_MAX;

    for (size_t i = b + tid; i < e; i += WARP_SIZE) {
        Key_T v = keys[i];
        for (size_t j = 0; j < n; ++j) {
            Count_T *h = hash_table + j * n;
            Hashed_T hv = hash(seed + j * s_sz, s_sz, v) % n;
            thread_min = min(thread_min, h[hv]);
        }

        for (int j = 16; j >= 1; j = j >> 1) {
            Count_T t = __shfl_down_sync(0xffffffff, thread_min, j);
            if (tid < j) {
                thread_min = min(thread_min, t);
            }
        }

        if (tid == 0) {
            count[i] = thread_min;
        }
    }
}



template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_threadfence(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
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

    // if (threadIdx.x == 0) {
    //     printf("%d\n", blockIdx.x);
    // }

    for (size_t i = b; i < e; i++) {
        __shared__ Count_T *h[WARP_SIZE];
        __shared__ Key_T v;
        __shared__ Hashed_T hash_mask[WARP_SIZE];

        if (tid == 0) {
            v = keys[i];
        }
__threadfence_block();
        if (tid < 3 * m) { // 3 * m <= WARP_SIZE
            hash_mask[tid] = hash(seed + tid * s_sz, s_sz, v);
            // printf("hash_mask %u\n", hash_mask[tid]);
            __threadfence_block();
            if (tid < m){
                h[tid] = hash_table + (hash_mask[tid] % n) * m * WARP_SIZE;    
                // hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 0x0000000f));
                hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 31));
                // printf("hash_mask %u\n", hash_mask[tid]);
            }
__threadfence_block();
        }
        unsigned int warp_mask = 1u << tid;
        for (size_t j = 0; j < m; ++j) {
            if (hash_mask[j] & warp_mask) {
                atomicAdd(h[j] + tid, 1);
            }
            
        }
    }
}

// #define HASH_MASK_MAX_SIZE (4 * 32)

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_low_threadfence(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
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
        __shared__ Count_T *h[WARP_SIZE];
        __shared__ Hashed_T hash_mask[sizeof(Count_T) * WARP_SIZE];

        h[tid] = hash_table + (hash(seed + tid * s_sz, s_sz, v) % n) * m * WARP_SIZE;

        for (size_t mid = tid; mid < 20 * m; mid += WARP_SIZE) {
            hash_mask[mid] = hash(seed + mid * s_sz, s_sz, v);
            // printf("mid: %lu\n", mid);
        }

        __threadfence_block();

        Hashed_T hm_one_bit = 0;
        Hashed_T hm_one_tid;
        if (tid < m) {
            hm_one_tid = hash_mask[tid] & 3u;
            hm_one_bit = (hash_mask[tid] >> 2) & 31u;
        }

        if (tid < 4 * m) {
            hash_mask[tid] = hash_mask[tid] & hash_mask[tid + (4 * m)] & hash_mask[tid + (8 * m)] & hash_mask[tid + (12 * m)] & hash_mask[tid + (16 * m)];
            
             // | (1u << (hash_mask[tid] & 31));
        }

        if (tid < m) {
            hash_mask[tid * 4 + hm_one_tid] |= (1u << (hm_one_bit));
        }

        __threadfence_block();

        unsigned int warp_mask = 1u << tid;
        for (size_t j = 0; j < m; ++j) {

            // printf("%u, %u, %u, %u\n", hash_mask[j * 4], hash_mask[j * 4 + 1], hash_mask[j * 4 + 2], hash_mask[j * 4 + 3]);

            Count_T add = 0;
            if (hash_mask[j * 4] & warp_mask) {
                add = add | (1u);
            }
            if (hash_mask[j * 4 + 1] & warp_mask) {
                add = add | (1u << 8);
            }
            if (hash_mask[j * 4 + 2] & warp_mask) {
                add = add | (1u << 16);
            }
            if (hash_mask[j * 4 + 3] & warp_mask) {
                add = add | (1u << 24);
            }

            // if (add == 0) {
            //     printf("err \n");
            // }

            if (add != 0) {
                atomicAdd(h[j] + tid, add);
            }

        }
    }
}


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_two_levels_threadfence_byte(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low, size_t m_low,
    Count_T *hash_table_high, size_t n_high, size_t m_high,
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
        __shared__ Count_T *h_low[WARP_SIZE];
        __shared__ Count_T *h_high[WARP_SIZE];
        __shared__ Hashed_T hash_mask[sizeof(Count_T) * WARP_SIZE];

        Hashed_T hv = hash(seed + tid * s_sz, s_sz, v);
        h_low[tid] = hash_table_low + (hv % n_low) * m_low * WARP_SIZE;
        h_high[tid] = hash_table_high + (hv % n_high) * m_high * WARP_SIZE;

        for (size_t mid = tid; mid < 20 * m_low; mid += WARP_SIZE) {
            hash_mask[mid] = hash(seed + mid * s_sz, s_sz, v);
            // printf("mid: %lu\n", mid);
        }

        __threadfence_block();

        Hashed_T hm_one_bit = 0;
        Hashed_T hm_one_tid;
        if (tid < m_low) {
            hm_one_tid = hash_mask[tid] & 3u;
            hm_one_bit = (hash_mask[tid] >> 2) & 31u;
        }

        if (tid < 4 * m_low) {
            hash_mask[tid] = hash_mask[tid] & hash_mask[tid + (4 * m_low)] & hash_mask[tid + (8 * m_low)] & hash_mask[tid + (12 * m_low)] & hash_mask[tid + (16 * m_low)];
            
             // | (1u << (hash_mask[tid] & 31));
        }

        if (tid < m_low) {
            hash_mask[tid * 4 + hm_one_tid] |= (1u << (hm_one_bit));
        }

        __threadfence_block();

        // unsigned int warp_mask = 1u << tid;
        // for (size_t j = 0; j < m; ++j) {
        //     Count_T add = 0;
        //     if (hash_mask[j * 4] & warp_mask) {
        //         add = add | (1u);
        //     }
        //     if (hash_mask[j * 4 + 1] & warp_mask) {
        //         add = add | (1u << 8);
        //     }
        //     if (hash_mask[j * 4 + 2] & warp_mask) {
        //         add = add | (1u << 16);
        //     }
        //     if (hash_mask[j * 4 + 3] & warp_mask) {
        //         add = add | (1u << 24);
        //     }

        //     if (add != 0) {
        //         atomicAdd(h[j] + tid, add);
        //     }

        // }

        unsigned int warp_mask = 1u << tid;
        Count_T thread_min = 0xffffffffu;
        Count_T thread_max = 0;
        for (size_t j = 0; j < m_low; ++j) {
            Count_T add = 0;
            Count_T cv = h_low[j][tid];
            if (hash_mask[j * 4] & warp_mask) {
                add = add | (1u);
                thread_min = min(thread_min, (cv & 0x000000ffu));
                thread_max = max(thread_max, (cv & 0x000000ffu));
            }
            if (hash_mask[j * 4 + 1] & warp_mask) {
                add = add | (1u << 8);
                thread_min = min(thread_min, (cv & 0x0000ff00u) >> 8);
                thread_max = max(thread_max, (cv & 0x0000ff00u) >> 8);
            }
            if (hash_mask[j * 4 + 2] & warp_mask) {
                add = add | (1u << 16);
                thread_min = min(thread_min, (cv & 0x00ff0000u) >> 16);
                thread_max = max(thread_max, (cv & 0x00ff0000u) >> 16);

            }
            if (hash_mask[j * 4 + 3] & warp_mask) {
                add = add | (1u << 24);
                thread_min = min(thread_min, (cv & 0xff000000u) >> 24);
                thread_max = max(thread_max, (cv & 0xff000000u) >> 24);
            }

            if (add != 0) {
                atomicAdd(h_low[j] + tid, add);
            }

        }

        for (int j = 16; j >= 1; j = j >> 1) {
            Count_T t_min = __shfl_down_sync(0xffffffff, thread_min, j);
            Count_T t_max = __shfl_down_sync(0xffffffff, thread_max, j);
            if (tid < j) {
                thread_min = min(thread_min, t_min);
                thread_max = max(thread_max, t_max);
            }
        }

        __shared__ bool update;
        // update = false;
        if (tid == 0) {
            // count[i] = thread_min;

            // if (thread_min >= 256) {
            //     printf("err\n");
            //     // exit(0);
            // }

            // if (thread_min > thread_max) {
            //     printf("err: thread_max: %u, thread_min: %u\n", thread_max, thread_min);
            // }

            // if (thread_max >= 32 || thread_min >= 32)
            if (thread_max >= 2 && thread_min > 0)
            {
                update = true;
            } else {
                update = false;
            }
        }

        __threadfence_block();

        if (update) {

            thread_min = __shfl_sync(0xffffffff, thread_min, 0);
            // printf("update thread_min: %u\n", thread_min);

            __shared__ Hashed_T hash_mask_high[WARP_SIZE];

            if (tid < 3 * m_high) { // 3 * m <= WARP_SIZE
                hash_mask_high[tid] = hash(seed + tid * s_sz, s_sz, v);
                __threadfence_block();
                if (tid < m_high){
                    // h[tid] = hash_table + (hash_mask[tid] % n) * m * WARP_SIZE;    
                    // hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 0x0000000f));
                    hash_mask_high[tid] = hash_mask_high[tid] & hash_mask_high[tid + m_high] & hash_mask_high[tid + (m_high << 1)] | (1u << (hash_mask_high[tid] & 31));
                    // printf("hash_mask %u\n", hash_mask[tid]);
                }
                __threadfence_block();
            }

            for (size_t j = 0; j < m_high; ++j)
            {
                if (hash_mask_high[j] & warp_mask) {
                    atomicAdd(h_high[j] + tid, thread_min);
                    // atomicAdd(h_high[j] + tid, 1);
                }
            }

            
            for (size_t j = 0; j < m_low; ++j) {
                Count_T add = 0;
                // Count_T add_high = 0;
                // Count_T cv = h[j][tid];
                if (hash_mask[j * 4] & warp_mask) {
                    add = add | (thread_min);
                    // add_high += thread_min;
                    // thread_min = min(thread_min, (cv & 0x000000ffu));
                }
                if (hash_mask[j * 4 + 1] & warp_mask) {
                    add = add | (thread_min << 8);
                    // thread_min = min(thread_min, (cv & 0x0000ff00u) >> 8);
                }
                if (hash_mask[j * 4 + 2] & warp_mask) {
                    add = add | (thread_min << 16);
                    // thread_min = min(thread_min, (cv & 0x00ff0000u) >> 16);
                }
                if (hash_mask[j * 4 + 3] & warp_mask) {
                    add = add | (thread_min << 24);
                    // thread_min = min(thread_min, (cv & 0xff000000u) >> 24);
                }

                if (add != 0) {
                    atomicSub(h_low[j] + tid, add);
                }
            }

            if (false) {
                Count_T thread_min_debug = 0xffffffffu;
                for (size_t j = 0; j < m_low; ++j) {
                    // Count_T add = 0;
                    // Count_T add_high = 0;
                    Count_T cv = h_low[j][tid];
                    if (hash_mask[j * 4] & warp_mask) {
                        // add = add | (thread_min);
                        // add_high += thread_min;
                        thread_min_debug = min(thread_min_debug, (cv & 0x000000ffu));
                    }
                    if (hash_mask[j * 4 + 1] & warp_mask) {
                        // add = add | (thread_min << 8);
                        thread_min_debug = min(thread_min_debug, (cv & 0x0000ff00u) >> 8);
                    }
                    if (hash_mask[j * 4 + 2] & warp_mask) {
                        // add = add | (thread_min << 16);
                        thread_min_debug = min(thread_min_debug, (cv & 0x00ff0000u) >> 16);
                    }
                    if (hash_mask[j * 4 + 3] & warp_mask) {
                        // add = add | (thread_min << 24);
                        thread_min_debug = min(thread_min_debug, (cv & 0xff000000u) >> 24);
                    }

                    // if (add != 0) {
                    //     atomicSub(h_low[j] + tid, add);
                    // }
                }

                for (int j = 16; j >= 1; j = j >> 1) {
                    Count_T t = __shfl_down_sync(0xffffffff, thread_min_debug, j);
                    if (tid < j) {
                        thread_min_debug = min(thread_min_debug, t);
                    }
                }

                if (tid == 0) {
                    // printf("thread_min: %u, thread_min_debug: %u\n", thread_min, thread_min_debug);
                }

            }
        }

        // for (size_t j = 0; j < m; ++j) {
        //     if (hash_mask[j] & warp_mask) {
        //         atomicAdd(h[j] + tid, 1);
        //     }
            
        // }
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_two_levels_threadfence(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low, size_t m_low,
    Count_T *hash_table_high, size_t n_high, size_t m_high,
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

    if (wid == 0 && tid == 0) {
        printf("insert_warp_two_levels_threadfence\n");
    }

    for (size_t i = b; i < e; i++) {
        __shared__ Count_T *h[WARP_SIZE];
        __shared__ Key_T v;
        __shared__ Hashed_T hash_mask[WARP_SIZE];

        if (tid == 0) {
            v = keys[i];
        }
__threadfence_block();
        if (tid < 3 * m_low) { // 3 * m <= WARP_SIZE
            hash_mask[tid] = hash(seed + tid * s_sz, s_sz, v);
            // printf("hash_mask %u\n", hash_mask[tid]);
            __threadfence_block();
            if (tid < m_low){
                h[tid] = hash_table_low + (hash_mask[tid] % n_low) * m_low * WARP_SIZE;    
                // hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 0x0000000f));
                hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m_low] & hash_mask[tid + (m_low << 1)] | (1u << (hash_mask[tid] & 31));
                // printf("hash_mask %u\n", hash_mask[tid]);
            }
__threadfence_block();
        }
        unsigned int warp_mask = 1u << tid;
        Count_T thread_min = 0xffffffffu;
        Count_T thread_max = 0;
        for (size_t j = 0; j < m_low; ++j) {
            if (hash_mask[j] & warp_mask) {
                Count_T v = h[j][tid];
                if (v < thread_min) { // it should be atomic
                    thread_min = v;
                }
                if (v > thread_max) {
                    thread_max = v;
                }
                // atomicMin(&thread_min, h[j][tid]);
            }
            
        }

        for (int j = 16; j >= 1; j = j >> 1) {
            Count_T t_min = __shfl_down_sync(0xffffffff, thread_min, j);
            Count_T t_max = __shfl_down_sync(0xffffffff, thread_max, j);
            if (tid < j) {
                thread_min = min(thread_min, t_min);
                thread_max = max(thread_max, t_max);
            }
        }

        __shared__ bool update;
        if (tid == 0) {

            if (thread_min >= 8) {
                update = true;
            } else {
                update = false;

            }

            
        }

        __threadfence_block();

        if (update) {
            thread_min = __shfl_sync(0xffffffff, thread_min, 0);

            __shared__ Count_T *h_high[WARP_SIZE];
            __shared__ Hashed_T hash_mask_high[WARP_SIZE];
            
            if (tid < 3 * m_high) { // 3 * m <= WARP_SIZE
                hash_mask_high[tid] = hash(seed + tid * s_sz, s_sz, v);
                // printf("hash_mask %u\n", hash_mask[tid]);
                __threadfence_block();
                if (tid < m_high){
                    h_high[tid] = hash_table_high + (hash_mask_high[tid] % n_high) * m_high * WARP_SIZE;    
                    // hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 0x0000000f));
                    hash_mask_high[tid] = hash_mask_high[tid] & hash_mask_high[tid + m_high] & hash_mask_high[tid + (m_high << 1)] | (1u << (hash_mask_high[tid] & 31));
                    // printf("hash_mask %u\n", hash_mask[tid]);
                }
    __threadfence_block();
            }

            for (size_t j = 0; j < m_high; ++j) {
            // size_t mid = j * WARP_SIZE + tid;
                if (hash_mask_high[j] & warp_mask) {
                    // atomicMax(h[j] + tid, thread_min);
                    // atomicAdd(h_high[j] + tid, thread_min + 1);
                    atomicAdd(h_high[j] + tid, 1);
                }
                
            }
            
            


            for (size_t j = 0; j < m_low; ++j) {
                // size_t mid = j * WARP_SIZE + tid;
                if (hash_mask[j] & warp_mask) {
                    // atomicMax(h[j] + tid, thread_min);
                    // atomicSub(h[j] + tid, thread_min);
                }
                
            }

        } else {
            // thread_min++;
            // thread_min = __shfl_sync(0xffffffff, thread_min, 0);

            for (size_t j = 0; j < m_low; ++j) {
                // size_t mid = j * WARP_SIZE + tid;
                if (hash_mask[j] & warp_mask) {
                    // atomicMax(h[j] + tid, thread_min);
                    atomicAdd(h[j] + tid, 1);
                }
                
            }
        }
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_min_threadfence(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
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

    for (size_t i = b; i < e; i++) {
        __shared__ Count_T *h[WARP_SIZE];
        __shared__ Key_T v;
        __shared__ Hashed_T hash_mask[WARP_SIZE];

        if (tid == 0) {
            v = keys[i];
        }
__threadfence_block();
        if (tid < 3 * m) { // 3 * m <= WARP_SIZE
            hash_mask[tid] = hash(seed + tid * s_sz, s_sz, v);
            // printf("hash_mask %u\n", hash_mask[tid]);
            __threadfence_block();
            if (tid < m){
                h[tid] = hash_table + (hash_mask[tid] % n) * m * WARP_SIZE;    
                // hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 0x0000000f));
                hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 31));
                // printf("hash_mask %u\n", hash_mask[tid]);
            }
__threadfence_block();
        }
        unsigned int warp_mask = 1u << tid;
        Count_T thread_min = 0xffffffffu;
        for (size_t j = 0; j < m; ++j) {
            if (hash_mask[j] & warp_mask) {
                if (h[j][tid] < thread_min) { // it should be atomic
                    thread_min = h[j][tid];
                }
                // atomicMin(&thread_min, h[j][tid]);
            }
            
        }

        for (int j = 16; j >= 1; j = j >> 1) {
            Count_T t = __shfl_down_sync(0xffffffff, thread_min, j);
            if (tid < j) {
                thread_min = min(thread_min, t);
            }
        }

        if (tid == 0) {
            thread_min++;
        }

        thread_min = __shfl_sync(0xffffffff, thread_min, 0);

        for (size_t j = 0; j < m; ++j) {
            // size_t mid = j * WARP_SIZE + tid;
            if (hash_mask[j] & warp_mask) {
                atomicMax(h[j] + tid, thread_min);
            }
            
        }

    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_warp_min(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
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

    for (size_t i = b; i < e; i++) {
        __shared__ Count_T *h[WARP_SIZE];
        __shared__ Key_T v;
        __shared__ Hashed_T hash_mask[WARP_SIZE];

        if (tid == 0) {
            v = keys[i];
        }
__threadfence_block();
        if (tid < 3 * m) { // 3 * m <= WARP_SIZE
            hash_mask[tid] = hash(seed + tid * s_sz, s_sz, v);
            __threadfence_block();
            if (tid < m){
                h[tid] = hash_table + (hash_mask[tid] % n) * m * WARP_SIZE;    
                // hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 0x0000000f));
                hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 31));
            }
__threadfence_block();
        }
        unsigned int warp_mask = 1u << tid;
        Count_T thread_min = 0xffffffffu;
        for (size_t j = 0; j < m; ++j) {
            // size_t mid = j * WARP_SIZE + tid;
            if (hash_mask[j] & warp_mask) {
                // thread_min = h[j][tid];
                if (h[j][tid] < thread_min) {
                    thread_min = h[j][tid];
                }
                // atomicMin(&thread_min, h[j][tid]);
            }
            
        }

        for (int j = 16; j >= 1; j = j >> 1) {
            Count_T t = __shfl_down_sync(0xffffffff, thread_min, j);
            if (tid < j) {
                thread_min = min(thread_min, t);
            }
        }

        if (tid == 0) {
            count[i] = thread_min;
        }
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_warp_low_min_threadfence(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
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

    // if (wid == 0 && tid == 0) {
    //     printf("search_warp_low_min_threadfence\n");
    // }
    for (size_t i = b; i < e; ++i) {
        // printf("search :%lu\n", i);
        Key_T v = keys[i];
        __shared__ Count_T *h[WARP_SIZE];
        __shared__ Hashed_T hash_mask[sizeof(Count_T) * WARP_SIZE];

        h[tid] = hash_table + (hash(seed + tid * s_sz, s_sz, v) % n) * m * WARP_SIZE;

        for (size_t mid = tid; mid < 20 * m; mid += WARP_SIZE) {
            hash_mask[mid] = hash(seed + mid * s_sz, s_sz, v);
        }

        __threadfence_block();

        Hashed_T hm_one_bit = 0;
        Hashed_T hm_one_tid;
        if (tid < m) {
            hm_one_tid = hash_mask[tid] & 3u;
            hm_one_bit = (hash_mask[tid] >> 2) & 31u;
        }

        if (tid < 4 * m) {
            hash_mask[tid] = hash_mask[tid] & hash_mask[tid + (4 * m)] & hash_mask[tid + (8 * m)] & hash_mask[tid + (12 * m)] & hash_mask[tid + (16 * m)];
            
             // | (1u << (hash_mask[tid] & 31));
        }

        if (tid < m) {
            hash_mask[tid * 4 + hm_one_tid] |= (1u << (hm_one_bit));
        }

        __threadfence_block();

        unsigned int warp_mask = 1u << tid;
        Count_T thread_min = 0xffffffffu;
        for (size_t j = 0; j < m; ++j) {
            // Count_T add = 0;
            Count_T cv = h[j][tid];
            if (hash_mask[j * 4] & warp_mask) {
                // add = add | (1u);
                thread_min = min(thread_min, (cv & 0x000000ffu));
            }
            if (hash_mask[j * 4 + 1] & warp_mask) {
                // add = add | (1u << 8);
                thread_min = min(thread_min, (cv & 0x0000ff00u) >> 8);
            }
            if (hash_mask[j * 4 + 2] & warp_mask) {
                // add = add | (1u << 16);
                thread_min = min(thread_min, (cv & 0x00ff0000u) >> 16);
            }
            if (hash_mask[j * 4 + 3] & warp_mask) {
                // add = add | (1u << 24);
                thread_min = min(thread_min, (cv & 0xff000000u) >> 24);
            }

        }

        for (int j = 16; j >= 1; j = j >> 1) {
            Count_T t = __shfl_down_sync(0xffffffff, thread_min, j);
            if (tid < j) {
                thread_min = min(thread_min, t);
            }
        }

        if (tid == 0) {
            count[i] = thread_min;
        }
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_warp_two_levels_threadfence_byte(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low, size_t m_low,
    Count_T *hash_table_high, size_t n_high, size_t m_high,
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



    for (size_t i = b; i < e; ++i) {
        Key_T v = keys[i];
        __shared__ Count_T *h_low[WARP_SIZE];
        __shared__ Count_T *h_high[WARP_SIZE];
        __shared__ Hashed_T hash_mask[sizeof(Count_T) * WARP_SIZE];

        Hashed_T hv = hash(seed + tid * s_sz, s_sz, v);
        h_low[tid] = hash_table_low + (hv % n_low) * m_low * WARP_SIZE;
        h_high[tid] = hash_table_high + (hv % n_high) * m_high * WARP_SIZE;

        for (size_t mid = tid; mid < 20 * m_low; mid += WARP_SIZE) {
            hash_mask[mid] = hash(seed + mid * s_sz, s_sz, v);
            // printf("mid: %lu\n", mid);
        }

        __threadfence_block();

        Hashed_T hm_one_bit = 0;
        Hashed_T hm_one_tid;
        if (tid < m_low) {
            hm_one_tid = hash_mask[tid] & 3u;
            hm_one_bit = (hash_mask[tid] >> 2) & 31u;
        }

        if (tid < 4 * m_low) {
            // hash_mask[tid] = hash_mask[tid] & hash_mask[tid + (4 * m)] & hash_mask[tid + (8 * m)] & hash_mask[tid + (12 * m)] & hash_mask[tid + (16 * m)];
            hash_mask[tid] = hash_mask[tid] & hash_mask[tid + (4 * m_low)] & hash_mask[tid + (8 * m_low)] & hash_mask[tid + (12 * m_low)] & hash_mask[tid + (16 * m_low)];

             // | (1u << (hash_mask[tid] & 31));
        }

        if (tid < m_low) {
            hash_mask[tid * 4 + hm_one_tid] |= (1u << (hm_one_bit));
        }

        __threadfence_block();

        unsigned int warp_mask = 1u << tid;
        Count_T thread_min = 0xffffffffu;
        Count_T thread_min_high = 0xffffffffu;


        // thread_min = __shfl_sync(0xffffffff, thread_min, 0);
        // printf("update thread_min: %u\n", thread_min);

        __shared__ Hashed_T hash_mask_high[WARP_SIZE];

        if (tid < 3 * m_high) { // 3 * m <= WARP_SIZE
            hash_mask_high[tid] = hash(seed + tid * s_sz, s_sz, v);
            __threadfence_block();
            if (tid < m_high){
                // h[tid] = hash_table + (hash_mask[tid] % n) * m * WARP_SIZE;    
                // hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 0x0000000f));
                hash_mask_high[tid] = hash_mask_high[tid] & hash_mask_high[tid + m_high] & hash_mask_high[tid + (m_high << 1)] | (1u << (hash_mask_high[tid] & 31));
                // printf("hash_mask %u\n", hash_mask[tid]);
            }
            __threadfence_block();
        }

        for (size_t j = 0; j < m_high; ++j)
        {
            if (hash_mask_high[j] & warp_mask) {
                // atomicAdd(h_high[j] + tid, thread_min);
                if (h_high[j][tid] < thread_min_high) {
                    thread_min_high = h_high[j][tid];
                }
            }
        }

        // for (size_t j = 0; j < m_high; ++j) {
        //     if (h_high[j][tid] < thread_min_high) {
        //         thread_min_high = h_high[j][tid];
        //     }
        // }
        

        for (size_t j = 0; j < m_low; ++j) {
            // Count_T add = 0;
            Count_T cv = h_low[j][tid];
            if (hash_mask[j * 4] & warp_mask) {
                // add = add | (1u);
                thread_min = min(thread_min, (cv & 0x000000ffu));
            }
            if (hash_mask[j * 4 + 1] & warp_mask) {
                // add = add | (1u << 8);
                thread_min = min(thread_min, (cv & 0x0000ff00u) >> 8);
            }
            if (hash_mask[j * 4 + 2] & warp_mask) {
                // add = add | (1u << 16);
                thread_min = min(thread_min, (cv & 0x00ff0000u) >> 16);
            }
            if (hash_mask[j * 4 + 3] & warp_mask) {
                // add = add | (1u << 24);
                thread_min = min(thread_min, (cv & 0xff000000u) >> 24);
            }

            // if (add != 0) {
            //     atomicAdd(h_low[j] + tid, add);
            // }

            

        }

        // thread_min = thread_min;

        for (int j = 16; j >= 1; j = j >> 1) {
            Count_T t = __shfl_down_sync(0xffffffff, thread_min, j);
            Count_T th = __shfl_down_sync(0xffffffff, thread_min_high, j);
            if (tid < j) {
                thread_min = min(thread_min, t);
                thread_min_high = min(thread_min_high, th);
            }
        }



        // __shared__ bool update = false;
        if (tid == 0) {
            // count[i] = thread_min + thread_min_high;
            count[i] = thread_min + thread_min_high;
            // printf("thread_min: %u, thread_min_high: %u\n", thread_min, thread_min_high);
            // if (thread_min >= 32) {
            //     update = true;
            // }
        }
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_warp_two_levels_threadfence(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low, size_t m_low,
    Count_T *hash_table_high, size_t n_high, size_t m_high,
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

    if (wid == 0 && tid == 0) {
        printf("search_warp_two_levels_threadfence\n");
    }

    for (size_t i = b; i < e; i++) {
        __shared__ Count_T *h[WARP_SIZE];
        __shared__ Key_T v;
        __shared__ Hashed_T hash_mask[WARP_SIZE];

        if (tid == 0) {
            v = keys[i];
        }
__threadfence_block();
        if (tid < 3 * m_low) { // 3 * m <= WARP_SIZE
            hash_mask[tid] = hash(seed + tid * s_sz, s_sz, v);
            // printf("hash_mask %u\n", hash_mask[tid]);
            __threadfence_block();
            if (tid < m_low){
                h[tid] = hash_table_low + (hash_mask[tid] % n_low) * m_low * WARP_SIZE;    
                // hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 0x0000000f));
                hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m_low] & hash_mask[tid + (m_low << 1)] | (1u << (hash_mask[tid] & 31));
                // printf("hash_mask %u\n", hash_mask[tid]);
            }
__threadfence_block();
        }
        unsigned int warp_mask = 1u << tid;
        Count_T thread_min = 0xffffffffu;
        Count_T thread_min_high = 0xffffffffu;
        // Count_T thread_max = 0;
        for (size_t j = 0; j < m_low; ++j) {
            if (hash_mask[j] & warp_mask) {
                Count_T v = h[j][tid];
                if (v < thread_min) { // it should be atomic
                    thread_min = v;
                }
                // if (v > thread_max) {
                //     thread_max = v;
                // }
                // atomicMin(&thread_min, h[j][tid]);
            }
            
        }



        // __shared__ bool update;
        // if (tid == 0) {

        //     if (thread_min >= 8) {
        //         update = true;
        //     } else {
        //         update = false;

        //     }

            
        // }

         __shared__ Count_T *h_high[WARP_SIZE];
        __shared__ Hashed_T hash_mask_high[WARP_SIZE];
        
        if (tid < 3 * m_high) { // 3 * m <= WARP_SIZE
            hash_mask_high[tid] = hash(seed + tid * s_sz, s_sz, v);
            // printf("hash_mask %u\n", hash_mask[tid]);
            __threadfence_block();
            if (tid < m_high){
                h_high[tid] = hash_table_high + (hash_mask_high[tid] % n_high) * m_high * WARP_SIZE;    
                // hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 0x0000000f));
                hash_mask_high[tid] = hash_mask_high[tid] & hash_mask_high[tid + m_high] & hash_mask_high[tid + (m_high << 1)] | (1u << (hash_mask_high[tid] & 31));
                // printf("hash_mask %u\n", hash_mask[tid]);
            }
__threadfence_block();
        }

        for (size_t j = 0; j < m_high; ++j) {
        // size_t mid = j * WARP_SIZE + tid;
            if (hash_mask_high[j] & warp_mask) {
                // atomicMax(h[j] + tid, thread_min);
                // atomicAdd(h_high[j] + tid, 1);
                Count_T v = h_high[j][tid];
                if (v < thread_min_high) { // it should be atomic
                    thread_min_high = v;
                }
            }
            
        }

        __threadfence_block();


        for (int j = 16; j >= 1; j = j >> 1) {
            Count_T t_min = __shfl_down_sync(0xffffffff, thread_min, j);
            Count_T t_min_high = __shfl_down_sync(0xffffffff, thread_min_high, j);
            // Count_T t_max = __shfl_down_sync(0xffffffff, thread_max, j);
            if (tid < j) {
                thread_min = min(thread_min, t_min);
                thread_min_high = min(thread_min_high, t_min_high);
                // thread_max = max(thread_max, t_max);
            }
        }

        if (tid == 0) {

            if (thread_min > 4000 ) {
                printf("err_thread_min: %u\n", thread_min);
            }
            if (thread_min_high > 4000 ) {
                printf("err_thread_min_high: %u\n", thread_min_high);
            }
            // || thread_min_high > 4000

            count[i] = thread_min + thread_min_high;
        }
    }
}