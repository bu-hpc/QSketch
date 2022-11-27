#pragma once

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
void insert_warp_host(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz, 
    size_t work_load_per_warp,
    void *debug)
{

    for (size_t i = 0; i < sz; ++i) {
        Count_T *h[WARP_SIZE];
        Key_T v = keys[i];
        Hashed_T hash_mask[WARP_SIZE];
        // std::cout << "i: " << i << std::endl;
        for (size_t tid = 0; tid < 3 * m; ++tid) {
            hash_mask[tid] = hash(seed + tid * s_sz, s_sz, v);
        }
        // std::cout << "p1" << std::endl;
        for (size_t tid = 0; tid < m; ++tid) {
            h[tid] = hash_table + (hash_mask[tid] % n) * m * WARP_SIZE;    
            hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 31));
        }
        // std::cout << "p2" << std::endl;
        for (size_t j = 0; j < m; ++j) {
            for (size_t tid = 0; tid < WARP_SIZE; ++tid) {
                unsigned int warp_mask = 1u << tid;
                if (hash_mask[j] & warp_mask) {
                    h[j][tid]++;
                }
            }
        }
        // std::cout << "p3" << std::endl;
        // std::cout << "i: " << i << std::endl;
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
void insert_warp_min_host(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz,
    size_t work_load_per_warp,
    void *debug)
{

    for (size_t i = 0; i < sz; ++i) {
        Count_T *h[WARP_SIZE];
        Key_T v = keys[i];
        Hashed_T hash_mask[WARP_SIZE];

        for (size_t tid = 0; tid < 3 * m; ++tid) {
            hash_mask[tid] = hash(seed + tid * s_sz, s_sz, v);

        }
        for (size_t tid = 0; tid < m; ++tid) {
            h[tid] = hash_table + (hash_mask[tid] % n) * m * WARP_SIZE;    
            hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 31));
        }
        Count_T thread_min = 0xffffffffu;
        for (size_t j = 0; j < m; ++j) {
            for (size_t tid = 0; tid < WARP_SIZE; ++tid) {
                unsigned int warp_mask = 1u << tid;
                if (hash_mask[j] & warp_mask) {
                    // h[j][tid]++;
                    if (h[j][tid] < thread_min) {
                        thread_min = h[j][tid];
                    }
                }
            }
        }

        for (size_t j = 0; j < m; ++j) {
            for (size_t tid = 0; tid < WARP_SIZE; ++tid) {
                unsigned int warp_mask = 1u << tid;
                if (hash_mask[j] & warp_mask) {
                    // h[j][tid]++;
                    if (h[j][tid] <= thread_min) {
                        h[j][tid]++;
                    }
                }
            }
        }
    }
}



template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
void search_warp_min_host(Key_T *keys, size_t sz,
    Count_T *hash_table, size_t n, size_t m,
    const Hash_Function &hash,
    Seed_T *seed, size_t s_sz,
    Count_T *count, Count_T Count_MAX,
    size_t work_load_per_warp,
    void *debug)
{

    for (size_t i = 0; i < sz; ++i) {
        Count_T *h[WARP_SIZE];
        Key_T v = keys[i];
        Hashed_T hash_mask[WARP_SIZE];

        for (size_t tid = 0; tid < 3 * m; ++tid) {
            hash_mask[tid] = hash(seed + tid * s_sz, s_sz, v);

        }
        for (size_t tid = 0; tid < m; ++tid) {
            h[tid] = hash_table + (hash_mask[tid] % n) * m * WARP_SIZE;    
            hash_mask[tid] = hash_mask[tid] & hash_mask[tid + m] & hash_mask[tid + (m << 1)] | (1u << (hash_mask[tid] & 31));
        }
        Count_T thread_min = Count_MAX;
        for (size_t j = 0; j < m; ++j) {
            for (size_t tid = 0; tid < WARP_SIZE; ++tid) {
                unsigned int warp_mask = 1u << tid;
                if (hash_mask[j] & warp_mask) {
                    // h[j][tid]++;
                    if (h[j][tid] < thread_min) {
                        thread_min = h[j][tid];
                    }
                }
            }
        }
        count[i] = thread_min;
        // std::cout << thread_min << std::endl;
    }
}

