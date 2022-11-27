#pragma once

// test 4:
// insert(key) :
//     lt <- low_level_table[hash(key) % n_low]
//     if (lt[31] != 0) :
//         insert_high(high_level_table[lt[31] * 128])
//     else 
//         if (max_count(lt, key) > 128) 
//             new_id = atomicAdd(mem_id, 1)
//             old = atomicMax(llt[31], new_id)                           : l0
//             if (old != 0) 
//                 while hlt_old[0][31] != 0 {}
//                 hlt_new = add(hlt_new, htl_old)
//             insert_high()
//         else 
//             insert_low()                                            : l2



// m_low == 1
// m_high == 1

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence(Key_T *keys, size_t sz,
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

    // if (threadIdx.x == 0) {
    //     printf("%d\n", blockIdx.x);
    // }

    // if (wid == 0) {
    //     printf("insert_warp_mem_threadfence\n");
    // }
    // return;

    // printf("%d, %d, %d\n", blockIdx.x, blockDim.x, threadIdx.x);
    // printf("wid %lu\n", wid);

    // if (wid != blockIdx.x) {

    // }
    // printf("err %lu, %d\n", wid, blockIdx.x);

    __shared__ Count_T *htl;
    __shared__ unsigned char hash_mask[16];

    for (size_t i = b; i < e; ++i) {
        Key_T v = keys[i];

        // cal_htl();
        Hashed_T hv = hash(seed + tid * s_sz, s_sz, v);
        if (tid == 0)
            htl = hash_table_low + (hv % n_low) * WARP_SIZE;
        
 
        // cal_hash_mask();

        unsigned char hash_bit = 0x0f;

        // #pragma unroll
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

        // if (tid == 0) {
        //     bool fd = false;
        //     for (size_t j = 0; j < 16; ++j) {
        //         if (hash_mask[j] != 0) {
        //             fd = true;
        //         }
        //     }
        //     if (!fd) {
        //         printf("err5\n");
        //     }
        // }
        

        unsigned int warp_mask_id = tid/8;
        unsigned char warp_mask = 1u << (tid % 8);
        if (htl[31] != 0) {
        // if (false) {
            // insert_high();
            // printf("err\n");
            Count_T *hth = hash_table_high + htl[31];
            
            atomicAdd(hth + 124, 1);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (hash_mask[i * 4 + warp_mask_id] & warp_mask) {
                    atomicAdd(hth + i * WARP_SIZE + tid, 1);
                }
            }
            atomicSub(hth + 124, 1);
            

        } else {
            // max_count = insert_low();
            // printf("eles\n");
            Count_T max_count = 0;
            Count_T add = 0;
            Count_T cv = htl[tid];
            unsigned char thm = (hash_mask[tid/2]);
            thm = thm >> ((tid % 2) * 4);
            // if (tid == 31 && thm != 0)
            //     printf("%u\n", thm & 0b00001111);
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
            // if (tid == 31 && add != 0) {
            //     printf("err2\n");
            // }
            if (add != 0)
                atomicAdd(htl + tid, add);

            for (int j = 16; j >= 1; j = j >> 1) {
                // Count_T t_min = __shfl_down_sync(0xffffffff, thread_min, j);
                Count_T t_max = __shfl_down_sync(0xffffffff, max_count, j);
                if (tid < j) {
                    // thread_min = min(thread_min, t_min);
                    max_count = max(max_count, t_max);
                }
            }

            __shared__ unsigned int old_id;
            __shared__ unsigned int new_id;

            if (tid == 0) {
                new_id = 0;
                old_id = 0;
                // printf("%u\n", max_count);
                if (max_count > 128) {
                    // upgrade
                    new_id = atomicAdd(mem_id, 1);
                    // printf("new_id: %u\n", new_id);
                    old_id = atomicMax(htl + 31, new_id);
                    if (old_id != 0) {
                        while (hash_table_high[old_id * 128 + 124] != 0) {}
                    }
                }
            }
            __threadfence_block();
            if (old_id != 0) {
                for (int i = tid; i < 128; i += WARP_SIZE) {
                    atomicAdd(hash_table_high + new_id * 128 + i, hash_table_high[old_id * 128 + i]);
                }
            }

            // if (max_count > 128) {
            //     new_id = atomicAdd(mem_id, 1);
            //     old_id = atomicMax(htl[j] + 31, new_id);
            //     if (old_id != 0) {
            //         while ( hash_table_high[old_idd * 128 + 31] != 0) {}
            //         for (int i = 0; i < 128; ++i) {
            //             atomicAdd(hash_table_high + new_id * 128 + i, hash_table_high[old_id * 128 + i]);
            //         }
            //     }
            // }
        }
    }
    // if (wid == 0 && tid == 0) {
    //     printf("kernel mem_id: %u\n", *mem_id);
    // }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_warp_mem_threadfence(Key_T *keys, size_t sz,
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

    // if (threadIdx.x == 0) {
    //     printf("%d\n", blockIdx.x);
    // }

    // if (wid == 0) {
    //     printf("insert_warp_mem_threadfence\n");
    // }
    // return;

    // printf("%d, %d, %d\n", blockIdx.x, blockDim.x, threadIdx.x);
    // printf("wid %lu\n", wid);

    // if (wid != blockIdx.x) {

    // }
    // printf("err %lu, %d\n", wid, blockIdx.x);

    __shared__ Count_T *htl;
    __shared__ unsigned char hash_mask[16];

    for (size_t i = b; i < e; ++i) {
        Key_T v = keys[i];

        // cal_htl();
        Hashed_T hv = hash(seed + tid * s_sz, s_sz, v);
        if (tid == 0)
            htl = hash_table_low + (hv % n_low) * WARP_SIZE;
        
 
        // cal_hash_mask();

        unsigned char hash_bit = 0x0f;

        // #pragma unroll
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

        unsigned int warp_mask_id = tid/8;
        unsigned char warp_mask = 1u << (tid % 8);

        Count_T thread_min_low = 0xffffffff;
        Count_T thread_min_high = 0xffffffff;

        {
            Count_T cv = htl[tid];
            unsigned char thm = (hash_mask[tid/2]);
            thm = thm >> ((tid % 2) * 4);
            // printf("%u\n", thm & 0b00001111);
            if (tid == 31 && thm != 0)
                printf("%u\n", thm & 0b00001111);
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

        }

        if (htl[31] != 0) {
            // insert_high();
            // printf("err3\n");
            Count_T *hth = hash_table_high + htl[31];
            
            // atomicAdd(hth + 124, 1);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (hash_mask[i * 4 + warp_mask_id] & warp_mask) {
                    // atomicAdd(hth + i * WARP_SIZE + tid, 1);
                    thread_min_high = min(thread_min_high, hth[i * WARP_SIZE + tid]);
                }
            }
            // atomicSub(hth + 124, 1);
            thread_min_low += thread_min_high;

        } 

        for (int j = 16; j >= 1; j = j >> 1) {
            Count_T t = __shfl_down_sync(0xffffffff, thread_min_low, j);
            if (tid < j) {
                thread_min_low = min(thread_min_low, t);
            }
        }

        if (tid == 0) {
            // if (thread_min_low > 102400) {
            //     printf("err4\n");
            // }
            count[i] = thread_min_low;
        }
    }
    // if (wid == 0 && tid == 0) {
    //     printf("kernel mem_id: %u\n", *mem_id);
    // }
}

// #define PRELOAD_SIZE 32

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_preload(Key_T *keys, size_t sz,
    Count_T *hash_table_low, size_t n_low,
    Count_T *hash_table_high, size_t n_high, unsigned int *mem_id,
    const Hash_Function &hash,
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

    // __shared__ Count_T *htl;
    // __shared__ unsigned char hash_mask[16];
    
    // // preload
    // Key_T pv;
    // Hashed_T phv;
    // __shared__ Count_T phtl_31;

    // {
    //     // insertion without preload
    //     size_t i = b;
    //     if (i < e) {
    //         Key_T v = keys[i];

    //         Hashed_T hv = hash(seed + tid * s_sz, s_sz, v);
    //         if (tid == 0) {
    //             htl = hash_table_low + (hv % n_low) * WARP_SIZE;
    //             htl_31 = htl[31];
    //         }
            
    //         unsigned char hash_bit = 0x0f;

    //         #pragma unroll
    //         for (int i = 0; i < 20; i += 4) {
    //             hash_bit &= (hv >> i);
    //         }

    //         unsigned char hash_bit_next = __shfl_down_sync(0xffffffff, hash_bit, 1);
    //         if (tid % 2 == 0) {
    //             hash_bit |= hash_bit_next << 4;
    //             hash_mask[tid/2] = hash_bit;
    //         }

    //         if (tid == 0) {
    //             Hashed_T hv_mask = hv % 124;
    //             hash_mask[hv_mask/8] |= 1 << (hv_mask % 8);
    //             hash_mask[15] &= 0b00001111;
    //         }

    //         unsigned int warp_mask_id = tid/8;
    //         unsigned char warp_mask = 1u << (tid % 8);
    //         if (phtl_31) {
    //         // if (htl_31) {
    //         // if (htl[31] != 0) {
    //         // if (false) {
    //             // insert_high();
    //             // printf("err\n");
    //             Count_T *hth = hash_table_high + htl[31];
                
    //             atomicAdd(hth + 124, 1);
    //             #pragma unroll
    //             for (int i = 0; i < 4; ++i) {
    //                 if (hash_mask[i * 4 + warp_mask_id] & warp_mask) {
    //                     atomicAdd(hth + i * WARP_SIZE + tid, 1);
    //                 }
    //             }
    //             atomicSub(hth + 124, 1);
                

    //         } else {
    //             // max_count = insert_low();
    //             // printf("eles\n");
    //             Count_T max_count = 0;
    //             Count_T add = 0;
    //             Count_T cv = htl[tid];
    //             unsigned char thm = (hash_mask[tid/2]);
    //             thm = thm >> ((tid % 2) * 4);
    //             // if (tid == 31 && thm != 0)
    //             //     printf("%u\n", thm & 0b00001111);
    //             if (thm & 0b00000001) {
    //                 add |= (1u);
    //                 max_count = max(max_count, (cv & 0x000000ffu));
    //             }

    //             if (thm & 0b00000010) {
    //                 add |= (1u << 8);
    //                 max_count = max(max_count, (cv & 0x0000ff00u) >> 8);
    //             }

    //             if (thm & 0b00000100) {
    //                 add |= (1u << 16);
    //                 max_count = max(max_count, (cv & 0x00ff0000u) >> 16);
    //             }

    //             if (thm & 0b00001000) {
    //                 add |= (1u << 24);
    //                 max_count = max(max_count, (cv & 0xff000000u) >> 24);
    //             }
    //             // if (tid == 31 && add != 0) {
    //             //     printf("err2\n");
    //             // }
    //             if (add != 0)
    //                 atomicAdd(htl + tid, add);

    //             for (int j = 16; j >= 1; j = j >> 1) {
    //                 // Count_T t_min = __shfl_down_sync(0xffffffff, thread_min, j);
    //                 Count_T t_max = __shfl_down_sync(0xffffffff, max_count, j);
    //                 if (tid < j) {
    //                     // thread_min = min(thread_min, t_min);
    //                     max_count = max(max_count, t_max);
    //                 }
    //             }

    //             __shared__ unsigned int old_id;
    //             __shared__ unsigned int new_id;

    //             if (tid == 0) {
    //                 new_id = 0;
    //                 old_id = 0;
    //                 // printf("%u\n", max_count);
    //                 if (max_count > 128) {
    //                     // upgrade
    //                     new_id = atomicAdd(mem_id, 1);
    //                     // printf("new_id: %u\n", new_id);
    //                     old_id = atomicMax(htl + 31, new_id);
    //                     if (old_id != 0) {
    //                         while (hash_table_high[old_id * 128 + 124] != 0) {}
    //                     }
    //                 }
    //             }
    //             __threadfence_block();
    //             if (old_id != 0) {
    //                 for (int i = tid; i < 128; i += WARP_SIZE) {
    //                     atomicAdd(hash_table_high + new_id * 128 + i, hash_table_high[old_id * 128 + i]);
    //                 }
    //             }

    //         }
    //         pv = v;
    //         phv = hv;
    //         phtl_31 = htl_31;
    //     }
    //     ++;
    // }

    // for (size_t i = b + 1; i < e; ++i) {
    //     Key_T v = keys[i];

    //     Hashed_T hv = hash(seed + tid * s_sz, s_sz, v);
    //     if (tid == 0) {
    //         htl = hash_table_low + (hv % n_low) * WARP_SIZE;
    //         htl_31 = htl[31];
    //     }
        
    //     unsigned char hash_bit = 0x0f;

    //     #pragma unroll
    //     for (int i = 0; i < 20; i += 4) {
    //         hash_bit &= (phv >> i);
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

    //     unsigned int warp_mask_id = tid/8;
    //     unsigned char warp_mask = 1u << (tid % 8);
    //     if (phtl_31) {
    //     // if (htl_31) {
    //     // if (htl[31] != 0) {
    //     // if (false) {
    //         // insert_high();
    //         // printf("err\n");
    //         // Count_T *hth = hash_table_high + htl[31];
    //         Count_T *hth = hash_table_high + phtl_31;

    //         atomicAdd(hth + 124, 1);
    //         #pragma unroll
    //         for (int i = 0; i < 4; ++i) {
    //             if (hash_mask[i * 4 + warp_mask_id] & warp_mask) {
    //                 atomicAdd(hth + i * WARP_SIZE + tid, 1);
    //             }
    //         }
    //         atomicSub(hth + 124, 1);
            

    //     } else {
    //         // max_count = insert_low();
    //         // printf("eles\n");
    //         Count_T max_count = 0;
    //         Count_T add = 0;
    //         Count_T cv = htl[tid];
    //         unsigned char thm = (hash_mask[tid/2]);
    //         thm = thm >> ((tid % 2) * 4);
    //         // if (tid == 31 && thm != 0)
    //         //     printf("%u\n", thm & 0b00001111);
    //         if (thm & 0b00000001) {
    //             add |= (1u);
    //             max_count = max(max_count, (cv & 0x000000ffu));
    //         }

    //         if (thm & 0b00000010) {
    //             add |= (1u << 8);
    //             max_count = max(max_count, (cv & 0x0000ff00u) >> 8);
    //         }

    //         if (thm & 0b00000100) {
    //             add |= (1u << 16);
    //             max_count = max(max_count, (cv & 0x00ff0000u) >> 16);
    //         }

    //         if (thm & 0b00001000) {
    //             add |= (1u << 24);
    //             max_count = max(max_count, (cv & 0xff000000u) >> 24);
    //         }
    //         // if (tid == 31 && add != 0) {
    //         //     printf("err2\n");
    //         // }
    //         if (add != 0)
    //             atomicAdd(htl + tid, add);

    //         for (int j = 16; j >= 1; j = j >> 1) {
    //             // Count_T t_min = __shfl_down_sync(0xffffffff, thread_min, j);
    //             Count_T t_max = __shfl_down_sync(0xffffffff, max_count, j);
    //             if (tid < j) {
    //                 // thread_min = min(thread_min, t_min);
    //                 max_count = max(max_count, t_max);
    //             }
    //         }

    //         __shared__ unsigned int old_id;
    //         __shared__ unsigned int new_id;

    //         if (tid == 0) {
    //             new_id = 0;
    //             old_id = 0;
    //             // printf("%u\n", max_count);
    //             if (max_count > 128) {
    //                 // upgrade
    //                 new_id = atomicAdd(mem_id, 1);
    //                 // printf("new_id: %u\n", new_id);
    //                 old_id = atomicMax(htl + 31, new_id);
    //                 if (old_id != 0) {
    //                     while (hash_table_high[old_id * 128 + 124] != 0) {}
    //                 }
    //             }
    //         }
    //         __threadfence_block();
    //         if (old_id != 0) {
    //             for (int i = tid; i < 128; i += WARP_SIZE) {
    //                 atomicAdd(hash_table_high + new_id * 128 + i, hash_table_high[old_id * 128 + i]);
    //             }
    //         }

    //         // if (max_count > 128) {
    //         //     new_id = atomicAdd(mem_id, 1);
    //         //     old_id = atomicMax(htl[j] + 31, new_id);
    //         //     if (old_id != 0) {
    //         //         while ( hash_table_high[old_idd * 128 + 31] != 0) {}
    //         //         for (int i = 0; i < 128; ++i) {
    //         //             atomicAdd(hash_table_high + new_id * 128 + i, hash_table_high[old_id * 128 + i]);
    //         //         }
    //         //     }
    //         // }
    //     }
    //     pv = v;
    //     phv = hv;
    //     phtl_31 = htl_31;
    // }
    // // if (wid == 0 && tid == 0) {
    // //     printf("kernel mem_id: %u\n", *mem_id);
    // // }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_test_kernel(Key_T *keys, size_t sz,
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

    if (threadIdx.x == 0) {
        printf("%d\n", blockIdx.x);
    }
}
