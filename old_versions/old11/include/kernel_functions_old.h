#pragma once


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_raw(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
    // Hashed_T (*hash)(Seed_T *, size_t, const Key_T &),
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{
    // printf("insert_warp\n");
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
__global__ void insert_warp(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
    // Hashed_T (*hash)(Seed_T *, size_t, const Key_T &),
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{
    // printf("insert_warp\n");
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e >= sz) {
        e = sz;
    }

    for (size_t i = b; i < e; i++) {
        __shared__ Count_T *h;
        __shared__ Key_T v;
        if (tid == 0) {
            v = keys[i];
            h = hash_table + (hash(seed, s_sz, v) % n) * m * WARP_SIZE;
        }

        for (size_t j = 0; j < m; ++j) {
            size_t mid = j * WARP_SIZE + tid;
            if (hash(seed + mid + 1, s_sz, v) % (7) == 1) {
                atomicAdd(h + mid, 1);
            }
        }
    }
}


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_test_1(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
    // Hashed_T (*hash)(Seed_T *, size_t, const Key_T &),
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
        __shared__ Count_T *h;
        __shared__ Key_T v;

        if (tid == 0) {
            v = keys[i];
            h = hash_table + (hash(seed, s_sz, v) % n) * m * WARP_SIZE;
        }
        
        // size_t l = (5 * m + WARP_SIZE - 1) / WARP_SIZE;
        size_t l = 4 * m;
        size_t m_table_size = m * WARP_SIZE;
        for (size_t j = tid; j < l; j += WARP_SIZE)
        {
            // size_t mid = j * WARP_SIZE + tid;
            Hashed_T hv = hash(seed + j, s_sz, v) % m_table_size;
            atomicAdd(h + hv, 1);
        }

        // for (size_t j = 0; j < m; ++j) {

        //     if (tid < 5) {
        //         size_t mid = j * WARP_SIZE + tid;
        //         Hashed_T hv = hash(seed + mid + 1, s_sz, v) % WARP_SIZE;
        //         atomicAdd(h + j * WARP_SIZE + hv, 1);
        //     }
            
        // }
    }
}


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_test_2(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
    // Hashed_T (*hash)(Seed_T *, size_t, const Key_T &),
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
        __shared__ Count_T *h;
        __shared__ Key_T v;
        __shared__ Hashed_T hash_mask[WARP_SIZE];

        if (tid == 0) {
            v = keys[i];
        }

        if (tid < 3 * m) { // 3 * m <= WARP_SIZE
            hash_mask[tid] = hash(seed + tid, s_sz, v);

            if (tid == 0) {
                h = hash_table + (hash_mask[tid] % n) * m * WARP_SIZE;               
            } 
            if (tid < m){ // ?
                hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + m << 1] | (1u << (hash_mask[tid] & 0x0000000f));
            }
        }
        unsigned int warp_mask = 1u << tid;
        for (size_t j = 0; j < m; ++j) {
            size_t mid = j * WARP_SIZE + tid;
            if (hash_mask[j] & warp_mask) {
                atomicAdd(h + mid, 1);
            }
            
        }
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_min(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
    // Hashed_T (*hash)(Seed_T *, size_t, const Key_T &),
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
        __shared__ Count_T *h;
        __shared__ Key_T v;
        __shared__ Hashed_T hash_mask[WARP_SIZE];

        if (tid == 0) {
            v = keys[i];
        }

        if (tid < 3 * m) { // 3 * m <= WARP_SIZE
            hash_mask[tid] = hash(seed + tid, s_sz, v);
__threadfence_block();

            if (tid == 0) {
                h = hash_table + (hash_mask[tid] % n) * m * WARP_SIZE;               
__threadfence_block();
            } 
            if (tid < m){
                hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 0x0000000f));
                // printf("%u\n", hash_mask[tid]);
                // printf("tid: %lu, %u\n", tid, hash_mask[tid]);
            }
__threadfence_block();

        }
        unsigned int warp_mask = 1u << tid;
        Count_T thread_min = 0xffffffffu;
        for (size_t j = 0; j < m; ++j) {
            size_t mid = j * WARP_SIZE + tid;
            if (hash_mask[j] & warp_mask) {
                // atomicAdd(h + mid, 1);
                // printf("%lu, %lu, %u, %lu, %lu\n", tid, j, hash_mask[j], h, mid);
                if (h[mid] < thread_min) {
                    thread_min = h[mid];
                }
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
            size_t mid = j * WARP_SIZE + tid;
            if (hash_mask[j] & warp_mask) {
                atomicMax(h + mid, thread_min);
                // if (h[mid] == 0) {
                //     printf("err: %lu\n", tid);
                // }
            }
            
        }

    }
}


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_min_2(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
    // Hashed_T (*hash)(Seed_T *, size_t, const Key_T &),
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{
    // thrust::device_vector<Hashed_T> &debug_hash_mask_count = *(thrust::device_vector<Hashed_T> *)(debug);
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
                // debug_hash_mask_count.push_
            }
__threadfence_block();
        }
        unsigned int warp_mask = 1u << tid;
        Count_T thread_min = 0xffffffffu;
        for (size_t j = 0; j < m; ++j) {
            // size_t mid = j * WARP_SIZE + tid;
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


// template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
// __global__ void insert_warp_min_3(Key_T *keys, size_t sz,
//     Count_T *hash_table, size_t n, size_t m,
//     // Hashed_T (*hash)(Seed_T *, size_t, const Key_T &),
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

//     for (size_t i = b; i < e; i++) {
//         // __shared__ Count_T *h[WARP_SIZE];
//         // __shared__ Key_T v;
//         // __shared__ Hashed_T hash_mask[WARP_SIZE];

//         // if (tid == 0) {
//         //     v = keys[i];
//         // }
//         Count_T *h;
//         Key_T v = keys[i];
//         Hashed_T hash_mask;
// // __threadfence_block();
//         if (tid < 3 * m) { // 3 * m <= WARP_SIZE
//             hash_mask = hash(seed + tid * s_sz, s_sz, v);
//             // __threadfence_block();
//             if (tid < m){
//                 h = hash_table + (hash_mask % n) * m * WARP_SIZE;    
//                 hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 0x0000000f));
//             }
// // __threadfence_block();
//         }
//         unsigned int warp_mask = 1u << tid;
//         Count_T thread_min = 0xffffffffu;
//         for (size_t j = 0; j < m; ++j) {
//             size_t mid = j * WARP_SIZE + tid;
//             if (hash_mask[j] & warp_mask) {
//                 if (h[j][mid] < thread_min) {
//                     thread_min = h[j][mid];
//                 }
//             }
            
//         }

//         for (int j = 16; j >= 1; j = j >> 1) {
//             Count_T t = __shfl_down_sync(0xffffffff, thread_min, j);
//             if (tid < j) {
//                 thread_min = min(thread_min, t);
//             }
//         }

//         if (tid == 0) {
//             thread_min++;
//         }

//         thread_min = __shfl_sync(0xffffffff, thread_min, 0);

//         for (size_t j = 0; j < m; ++j) {
//             size_t mid = j * WARP_SIZE + tid;
//             if (hash_mask[j] & warp_mask) {
//                 atomicMax(h[j] + mid, thread_min);
//             }
            
//         }

//     }
// }



template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_test_2_old2(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
    // Hashed_T (*hash)(Seed_T *, size_t, const Key_T &),
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
        __shared__ Count_T *h;
        __shared__ Key_T v;
        __shared__ Hashed_T hash_mask[WARP_SIZE];

        if (tid == 0) {
            v = keys[i];
        }

        if (tid <= m) {
            Hashed_T hv = hash(seed + tid, s_sz, v);

            if (tid == 0) {
                h = hash_table + (hv % n) * m * WARP_SIZE;               
            } else {
                hash_mask[tid - 1] = hv & circular_shift_l(hv, 1) & circular_shift_l(hv, 2) & (1u << (hv & 0x0000000f));
            }

            // hash_mask[tid] =  hash(seed + tid + 1, s_sz, v);
            // hash_mask[tid] = hash_mask[tid] & circular_shift_l(hash_mask[tid], 1) & circular_shift_l(hash_mask[tid], 2);
            // hash_mask[tid] = hash_mask[tid] & (1u << (hash_mask[tid] & 0x0000000f));
        }
        unsigned int warp_mask = 1u << tid;
        for (size_t j = 0; j < m; ++j) {
            size_t mid = j * WARP_SIZE + tid;
            if (hash_mask[j] & warp_mask) {
                atomicAdd(h + mid, 1);
            }
            
        }
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_test_2_old(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
    // Hashed_T (*hash)(Seed_T *, size_t, const Key_T &),
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
        __shared__ Count_T *h;
        __shared__ Key_T v;

        if (tid == 0) {
            v = keys[i];
            h = hash_table + (hash(seed, s_sz, v) % n) * m * WARP_SIZE;
        }
        __shared__ Hashed_T hash_mask[WARP_SIZE];
        if (tid < m) {
            hash_mask[tid] =  hash(seed + tid + 1, s_sz, v);
            hash_mask[tid] = hash_mask[tid] & circular_shift_l(hash_mask[tid], 1) & circular_shift_l(hash_mask[tid], 2);
            hash_mask[tid] = hash_mask[tid] & (1u << (hash_mask[tid] & 0x0000000f));
        }
        unsigned int warp_mask = 1u << tid;
        for (size_t j = 0; j < m; ++j) {
            size_t mid = j * WARP_SIZE + tid;
            if (hash_mask[j] & warp_mask) {
                atomicAdd(h + mid, 1);
            }
            
        }
    }
}

// #define Count_MAX 65536

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function_T>
__global__ void insert_min_warp(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
    // Hashed_T (*hash)(Seed_T *, size_t, const Key_T &),
    Hash_Function_T hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{
    // size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    // size_t tid = threadIdx.x % WARP_SIZE;
    // size_t b = wid * work_load_per_warp;
    // size_t e = (wid + 1) * work_load_per_warp;

    // if (e >= sz) {
    //     e = sz;
    // }

    // for (size_t i = b; i < e; i++) {
    //     __shared__ Count_T *h;
    //     __shared__ Key_T v;
    //     if (tid == 0) {
    //         v = keys[i];
    //         h = hash_table + (hash(seed, s_sz, v) % n) * m * WARP_SIZE;
    //     }

    //     Count_T thread_min = Count_MAX;
    //     for (size_t j = 0; j < m; ++j) {
    //         size_t mid = j * WARP_SIZE + tid;
    //         if (hash(seed + mid + 1, s_sz, v) % (7) == 1) {
    //             // atomicAdd(h + mid, 1);
    //             thread_min = min(thread_min, h[mid]);
    //         }
    //     }
    //     for (int j = 16; j >= 1; j /= 2) {
    //         Count_T t = __shfl_down_sync(0xffffffff, thread_min, j);
    //         if (tid < j) {
    //             thread_min = min(thread_min, t);
    //         }
    //     }

    //     __shared__ Count_T min_count;
    //     if (tid == 0) {
    //         min_count = thread_min;
    //     }

    //     for (size_t j = 0; j < m; ++j) {
    //         size_t mid = j * WARP_SIZE + tid;
    //         if (hash(seed + mid + 1, s_sz, v) % (7) == 1) {
    //             // atomicAdd(h + mid, 1);
    //             thread_min = min(thread_min, h[mid]);
    //         }
    //     }

    // }
}


// __global__ void insert2(T *keys, size_t sz,  size_t work_load_per_warp,
//     C *hash_table, size_t n, size_t m,
//     T *seed, size_t s_sz, 
//     void *debug)
// {TTT
//     size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
//     size_t tid = threadIdx.x % WARP_SIZE;
//     size_t b = wid * work_load_per_warp;
//     size_t e = (wid + 1) * work_load_per_warp;

//     for (size_t i = b; i < e; i += WARP_SIZE) {
//         T k = keys[i + tid];
//         size_t h = (hash2(seed, s_sz, k) % n) * m * WARP_SIZE;

//         for (size_t w = 0; w < WARP_SIZE; ++w) {

//             T kt = __shfl_sync(0xffffffff, k, w);
//             size_t ht = __shfl_sync(0xffffffff, h, w);

//             for (size_t j = 0; j < m; ++j) {
//                 size_t mid = j * WARP_SIZE + tid;
//                 // hash2(seed + mid + 1, s_sz, kt/*keys[t]*/);
//                 // atomicAdd(ht + mid, 1);
//                 if (hash2(seed + mid + 1, s_sz, kt/*keys[t]*/) % (7) == 1) {
//                     atomicAdd(hash_table + ht + mid, 1);
//                 }
//             }
//         }
        
//     }
// }

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_warp(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
    // Hashed_T (*hash)(Seed_T *, size_t, const Key_T &),
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
    for (size_t i = b; i < e; i++ ) {
        __shared__ Key_T *h;

        Key_T v = keys[i];
        if (tid == 0) {
            h = hash_table + (hash(seed, s_sz, v) % n) * m * WARP_SIZE;
        }

        __shared__ Count_T min[WARP_SIZE];
        min[tid] = Count_MAX;

        for (size_t j = 0; j < m; ++j) {
            size_t mid = j * WARP_SIZE + tid;
            if (hash(seed + mid + 1, s_sz, v) % (7) == 1) {
                if (h[mid] < min[tid]) {
                    min[tid] = h[mid];
                }
            }
        }
        Count_T ans = Count_MAX;
        if (tid == 0) {
            for (size_t j = 0; j < WARP_SIZE; ++j) {
                if (min[j] < ans) {
                    ans = min[j];
                }
            }
            count[i] = ans;
        }
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_warp_test_2(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
    // Hashed_T (*hash)(Seed_T *, size_t, const Key_T &),
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
        __shared__ Count_T *h;
        __shared__ Key_T v;
        __shared__ Hashed_T hash_mask[WARP_SIZE];

        if (tid == 0) {
            v = keys[i];
        }

        if (tid < 3 * m) { // 3 * m <= WARP_SIZE
            hash_mask[tid] = hash(seed + tid, s_sz, v);
__threadfence_block();
            if (tid == 0) {
                h = hash_table + (hash_mask[tid] % n) * m * WARP_SIZE;               
            } 
__threadfence_block();
            if (tid < m){
                hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 0x0000000f));
                // printf("%u\n", hash_mask[tid]);
                // printf("tid: %lu, %u\n", tid, hash_mask[tid]);
            }
__threadfence_block();
        }
        unsigned int warp_mask = 1u << tid;
        Count_T thread_min = 0xffffffffu;
        for (size_t j = 0; j < m; ++j) {
            size_t mid = j * WARP_SIZE + tid;
            if (hash_mask[j] & warp_mask) {
                // atomicAdd(h + mid, 1);
                // if (h[mid] == 0) {
                //     printf("err: %lu\n", tid);
                // }
                // printf("%lu, %lu, %u, %lu, %lu\n", tid, j, hash_mask[j], h, mid);
                if (h[mid] < thread_min) {
                    thread_min = h[mid];
                }
            }
            
        }

        // if (thread_min == 0) {
        //     printf("err: %lu\n", tid);
        // }
        for (int j = 16; j >= 1; j = j >> 1) {
            Count_T t = __shfl_down_sync(0xffffffff, thread_min, j);
            if (tid < j) {
                thread_min = min(thread_min, t);
            }
        }

        if (tid == 0) {
            count[i] = thread_min;
            // printf("thread_min: %u\n", thread_min);
        }
    }
}


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_warp_test_min_2(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
    // Hashed_T (*hash)(Seed_T *, size_t, const Key_T &),
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



// template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function_T>
// __global__ void search_warp(Key_T *keys, size_t sz,
//     Count_T *hash_table, size_t n, size_t m,
//     // Hashed_T (*hash)(Seed_T *, size_t, const Key_T &),
//     Hash_Function_T hash,
//     Seed_T *seed, size_t s_sz,
//     Count_T *count, Count_T Count_MAX,
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
//     for (size_t i = b; i < e; i++ ) {
//         __shared__ Key_T *h;

//         Key_T v = keys[i];
//         if (tid == 0) {
//             h = hash_table + (hash(seed, s_sz, v) % n) * m * WARP_SIZE;
//         }

//         __shared__ Count_T min[WARP_SIZE];
//         min[tid] = Count_MAX;

//         for (size_t j = 0; j < m; ++j) {
//             size_t mid = j * WARP_SIZE + tid;
//             if (hash(seed + mid + 1, s_sz, v) % (7) == 1) {
//                 if (h[mid] < min[tid]) {
//                     min[tid] = h[mid];
//                 }
//             }
//         }
//         Count_T ans = Count_MAX;
//         if (tid == 0) {
//             for (size_t j = 0; j < WARP_SIZE; ++j) {
//                 if (min[j] < ans) {
//                     ans = min[j];
//                 }
//             }
//             count[i] = ans;
//         }
//     }
// }