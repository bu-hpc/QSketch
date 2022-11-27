#pragma once

template <typename Key_T, typename Count_T, typename Hash_Function_T, typename Seed_T>
__global__ void insert_warp(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
    // Hashed_T (*hash)(Seed_T *, size_t, const Key_T &),
    Hash_Function_T hash,
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
    // printf("insert_warp: %d, %d\n", blockIdx.x, threadIdx.x);
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("blockIdx.x : %d\n", blockIdx.x);
    // }

    // printf(" tid : %lu\n", tid);

    for (size_t i = b; i < e; i++) {
        // printf("i : %lu\n", i);
        __shared__ Count_T *h;
        __shared__ Key_T v;
        if (tid == 0) {
            // printf("insert_warp i %lu, %lu\n", i, keys[i]);
            v = keys[i];
            // hash(seed, s_sz, v);

            // auto hv = hash(seed, s_sz, v);
            // printf("i : %lu, hv: %u\n", i, hv);

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
