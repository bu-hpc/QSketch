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

template <typename Count_T>
__global__ void init_hash_table_high(Count_T *hash_table_high, size_t sz)
{
    size_t tid = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t tnum = gridDim.x * blockDim.x;
    // if (tid == 0) {
    //     printf("tnum: %lu\n", tnum);
    // }

    for (; tid * 128 + 124 < sz; tid += tnum) {
        hash_table_high[tid * 128 + 124] = 1;
    }
}



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

    // __shared__ Count_T *htl;
    __shared__ unsigned char hash_mask[16];

    for (size_t i = b; i < e; ++i) {
        Key_T v = keys[i];

        // cal_htl();
        Hashed_T hv = hash(seed + tid * s_sz, s_sz, v);
        Hashed_T htl_offset;
        if (tid == 0) {
            htl_offset = (hv % n_low) * WARP_SIZE;
            // htl = hash_table_low + (hv % n_low) * WARP_SIZE;
        }
        htl_offset = __shfl_sync(0xffffffff, htl_offset, 0);
        Count_T *htl = hash_table_low + htl_offset;
// htl = __shfl_sync(0xffffffff, htl, 0);
 
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

        
        __threadfence_block();
        unsigned int warp_mask_id = tid/8;
        unsigned char warp_mask = 1u << (tid % 8);
        // Count_T h1 = htl[31];
        if (htl[31] > BUFFER_START) {
            // if (htl[31] > 9216) {
            //     printf("t1 err11\n");
            // }
            // if ((htl[31] - BUFFER_START) * 128 > 1024 * 1024) {
            //     if (tid == 0)
            //         printf("t1 err12\n");
            // }
        // if (false) {
            // insert_high();
            // printf("insert_high\n");
            Count_T *hth = hash_table_high + (htl[31] - BUFFER_START) * 128;

            // if (tid == 0)
            //     atomicAdd(hth + 124, 1);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (hash_mask[i * 4 + warp_mask_id] & warp_mask) {
                    // if (i * WARP_SIZE + tid >= 124) {
                    //     printf("err7\n");
                    // }
                    atomicAdd(hth + i * WARP_SIZE + tid, 1);
                }
            }
            // __threadfence();
            // if (tid == 0)
            //     atomicSub(hth + 124, 1);
            

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
            // if (add != 0)
            //     atomicAdd(htl + tid, add);

            for (int j = 16; j >= 1; j = j >> 1) {
                // Count_T t_min = __shfl_down_sync(0xffffffff, thread_min, j);
                Count_T t_max = __shfl_down_sync(0xffffffff, max_count, j);
                if (tid < j) {
                    // thread_min = min(thread_min, t_min);
                    max_count = max(max_count, t_max);
                }
            }

            max_count = __shfl_sync(0xffffffff, max_count, 0);
            // atomicAdd(htl + tid, add);

            if (add != 0 && max_count <= 128) {

                atomicAdd(htl + tid, add);
            }

            if (max_count > 128) 
            {

                unsigned int id = 0;
                unsigned int tem = 0;
                if (tid == 0) {
                    unsigned int old = atomicCAS(htl + 31, 0, 1);
                    // int trace = 0;
                    // volatile Count_T *htl_val = htl + 31;
                    if (old == 0) {
                        id = atomicAdd(mem_id, 1);
                        htl[31] = id;
                        // trace = 1;
                    } else {
                        // while (__ldcv(htl + 31) <= 1) {}
                        // tem = __ldcv(htl + 31);
                        // id = __ldcv(htl + 31);
                        while (id <= 1) {
                            // id = max(id, __ldcv(htl + 31));
                            id = __ldcv(htl + 31);
                            // id = *htl_val;
                            __threadfence_block();
                        }
                        // trace = 2;
                    }
                    // __threadfence_block();
                    // if (id <= BUFFER_START) {
                    //     printf("err14: %u, %d, %u, %u\n", id, trace, __ldcv(htl + 31), tem);
                    // }
                }
                __threadfence_block();
                id = __shfl_sync(0xffffffff, id, 0);
                // __threadfence_block();

                // if ((id - BUFFER_START) * 128 > 1024 * 1024) {
                //     if (tid == 0)
                //         printf("err12: %u, %u\n", id, htl[31]);
                // }

                Count_T *hth = hash_table_high + (id - BUFFER_START) * 128;

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (hash_mask[i * 4 + warp_mask_id] & warp_mask) {
                        atomicAdd(hth + i * WARP_SIZE + tid, 1);
                    }
                }

            } 
            
        }
    }
    
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_old3(Key_T *keys, size_t sz,
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

    // __shared__ Count_T *htl;
    __shared__ unsigned char hash_mask[16];

    for (size_t i = b; i < e; ++i) {
        Key_T v = keys[i];

        // cal_htl();
        Hashed_T hv = hash(seed + tid * s_sz, s_sz, v);
        Hashed_T htl_offset;
        if (tid == 0) {
            htl_offset = (hv % n_low) * WARP_SIZE;
            // htl = hash_table_low + (hv % n_low) * WARP_SIZE;
        }
        htl_offset = __shfl_sync(0xffffffff, htl_offset, 0);
        Count_T *htl = hash_table_low + htl_offset;
// htl = __shfl_sync(0xffffffff, htl, 0);
 
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
        
        __threadfence_block();
        unsigned int warp_mask_id = tid/8;
        unsigned char warp_mask = 1u << (tid % 8);
        // Count_T h1 = htl[31];
        if (htl[31] > BUFFER_START) {
            // if (htl[31] > 9216) {
            //     printf("t1 err11\n");
            // }
            // if ((htl[31] - BUFFER_START) * 128 > 1024 * 1024) {
            //     if (tid == 0)
            //         printf("t1 err12\n");
            // }
        // if (false) {
            // insert_high();
            // printf("insert_high\n");
            Count_T *hth = hash_table_high + (htl[31] - BUFFER_START) * 128;

            // if (tid == 0)
            //     atomicAdd(hth + 124, 1);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (hash_mask[i * 4 + warp_mask_id] & warp_mask) {
                    // if (i * WARP_SIZE + tid >= 124) {
                    //     printf("err7\n");
                    // }
                    atomicAdd(hth + i * WARP_SIZE + tid, 1);
                }
            }
            // __threadfence();
            // if (tid == 0)
            //     atomicSub(hth + 124, 1);
            

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
            // if (add != 0)
            //     atomicAdd(htl + tid, add);

            for (int j = 16; j >= 1; j = j >> 1) {
                // Count_T t_min = __shfl_down_sync(0xffffffff, thread_min, j);
                Count_T t_max = __shfl_down_sync(0xffffffff, max_count, j);
                if (tid < j) {
                    // thread_min = min(thread_min, t_min);
                    max_count = max(max_count, t_max);
                }
            }

            max_count = __shfl_sync(0xffffffff, max_count, 0);
            if (add != 0 && max_count <= 128) {

                atomicAdd(htl + tid, add);
            }

            /*
int atomicCAS(int* address, int compare, int val);
unsigned int atomicCAS(unsigned int* address,
                       unsigned int compare,
                       unsigned int val);
unsigned long long int atomicCAS(unsigned long long int* address,
                                 unsigned long long int compare,
                                 unsigned long long int val);
unsigned short int atomicCAS(unsigned short int *address, 
                             unsigned short int compare, 
                             unsigned short int val);
reads the 16-bit, 32-bit or 64-bit word old located at the address address in global or shared memory, 
computes (old == compare ? val : old) , and stores the result back to memory at the same address. 
These three operations are performed in one atomic transaction. The function returns old (Compare And Swap).

            */

            if (max_count > 128) {

                unsigned int id = 0;
                unsigned int tem = 0;
                if (tid == 0) {
                    unsigned int old = atomicCAS(htl + 31, 0, 1);
                    int trace = 0;
                    if (old == 0) {
                        id = atomicAdd(mem_id, 1);
                        htl[31] = id;
                        trace = 1;
                    } else {
                        // while (__ldcv(htl + 31) <= 1) {}
                        // tem = __ldcv(htl + 31);
                        // id = __ldcv(htl + 31);
                        while (id <= 1) {
                            // id = __ldcv(htl + 31);
                            __threadfence();
                        }
                        trace = 2;
                    }
                    __threadfence();
                    if (id <= BUFFER_START) {
                        // printf("err14: %u, %d, %u, %u\n", id, trace, __ldcv(htl + 31), tem);
                    }
                }
                __threadfence();
                id = __shfl_sync(0xffffffff, id, 0);
                __threadfence();
                // unsigned int id;
                // if (tid == 0) {
                //     id = atomicAdd(htl + 31, 1);
                //     if (id > 1024 * 1024 * 1024) {
                //         printf("err13\n");
                //     }
                //     if (id != 0) {
                //         while ( (id = __ldcv(htl + 31)) <= BUFFER_START) {}
                //     } else {
                //         htl[31] = id = atomicAdd(mem_id, 1);
                //     }
                // }
                // __threadfence();
                // id = __shfl_sync(0xffffffff, id, 0);

                // __threadfence();
                // if (htl[31] > 9216) {
                //     printf("err11\n");
                // }

                // if ((htl[31] - 1024) * 128 > 1024 * 1024) {
                //     if (tid == 0)
                //         printf("err12: %u\n", htl[31]);
                // }
                // if (id > 9216) {
                //     printf("err11\n");
                // }

                if ((id - BUFFER_START) * 128 > 1024 * 1024) {
                    if (tid == 0)
                        printf("err12: %u, %u\n", id, htl[31]);
                }

                Count_T *hth = hash_table_high + (id - BUFFER_START) * 128;

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (hash_mask[i * 4 + warp_mask_id] & warp_mask) {
                        atomicAdd(hth + i * WARP_SIZE + tid, 1);
                    }
                }

            } 
            
        }
    }
    
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_old2(Key_T *keys, size_t sz,
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

    // __shared__ Count_T *htl;
    __shared__ unsigned char hash_mask[16];

    for (size_t i = b; i < e; ++i) {
        Key_T v = keys[i];

        // cal_htl();
        Hashed_T hv = hash(seed + tid * s_sz, s_sz, v);
        Hashed_T htl_offset;
        if (tid == 0) {
            htl_offset = (hv % n_low) * WARP_SIZE;
            // htl = hash_table_low + (hv % n_low) * WARP_SIZE;
        }
        htl_offset = __shfl_sync(0xffffffff, htl_offset, 0);
        Count_T *htl = hash_table_low + htl_offset;
// htl = __shfl_sync(0xffffffff, htl, 0);
 
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
        
        __threadfence_block();
        unsigned int warp_mask_id = tid/8;
        unsigned char warp_mask = 1u << (tid % 8);
        // Count_T h1 = htl[31];
        if (htl[31] > BUFFER_START) {
            // if (htl[31] > 9216) {
            //     printf("t1 err11\n");
            // }
            if ((htl[31] - BUFFER_START) * 128 > 1024 * 1024) {
                if (tid == 0)
                    printf("t1 err12\n");
            }
        // if (false) {
            // insert_high();
            // printf("insert_high\n");
            Count_T *hth = hash_table_high + (htl[31] - BUFFER_START) * 128;

            // if (tid == 0)
            //     atomicAdd(hth + 124, 1);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (hash_mask[i * 4 + warp_mask_id] & warp_mask) {
                    // if (i * WARP_SIZE + tid >= 124) {
                    //     printf("err7\n");
                    // }
                    atomicAdd(hth + i * WARP_SIZE + tid, 1);
                }
            }
            // __threadfence();
            // if (tid == 0)
            //     atomicSub(hth + 124, 1);
            

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
            // if (add != 0)
            //     atomicAdd(htl + tid, add);

            for (int j = 16; j >= 1; j = j >> 1) {
                // Count_T t_min = __shfl_down_sync(0xffffffff, thread_min, j);
                Count_T t_max = __shfl_down_sync(0xffffffff, max_count, j);
                if (tid < j) {
                    // thread_min = min(thread_min, t_min);
                    max_count = max(max_count, t_max);
                }
            }

            max_count = __shfl_sync(0xffffffff, max_count, 0);
            if (add != 0 && max_count <= 128) {

                atomicAdd(htl + tid, add);
            }

            if (max_count > 128) {
                unsigned int id;
                if (tid == 0) {
                    id = atomicAdd(htl + 31, 1);
                    if (id > 1024 * 1024 * 1024) {
                        printf("err13\n");
                    }
                    if (id != 0) {
                        // while ( (id = __ldcv(htl + 31)) <= BUFFER_START) {}
                    } else {
                        htl[31] = id = atomicAdd(mem_id, 1);
                    }
                }
                __threadfence();
                id = __shfl_sync(0xffffffff, id, 0);

                __threadfence();
                // if (htl[31] > 9216) {
                //     printf("err11\n");
                // }

                // if ((htl[31] - 1024) * 128 > 1024 * 1024) {
                //     if (tid == 0)
                //         printf("err12: %u\n", htl[31]);
                // }
                // if (id > 9216) {
                //     printf("err11\n");
                // }

                if ((id - BUFFER_START) * 128 > 1024 * 1024) {
                    if (tid == 0)
                        printf("err12: %u, %u\n", id, htl[31]);
                }

                Count_T *hth = hash_table_high + (id - BUFFER_START) * 128;

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (hash_mask[i * 4 + warp_mask_id] & warp_mask) {
                        atomicAdd(hth + i * WARP_SIZE + tid, 1);
                    }
                }

            } 
            
        }
    }
    
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_warp_mem_threadfence_old(Key_T *keys, size_t sz,
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
        // Count_T h1 = htl[31];
        if (htl[31] != 0) {
        // if (false) {
            // insert_high();
            // printf("insert_high\n");
            Count_T *hth = hash_table_high + htl[31] * 128;
            // Count_T h2 = htl[31];
            // if (h1 != h2) {
            //     printf("err9\n");
            // }

            // Count_T h1_t0 = __shfl_sync(0xffffffff, h1, 0);
            // if (h1_t0 != h1) {
            //     printf("err10\n");
            // }

            if (tid == 0)
                atomicAdd(hth + 124, 1);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (hash_mask[i * 4 + warp_mask_id] & warp_mask) {
                    // if (i * WARP_SIZE + tid >= 124) {
                    //     printf("err7\n");
                    // }
                    atomicAdd(hth + i * WARP_SIZE + tid, 1);
                }
            }
            __threadfence();
            if (tid == 0)
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
            // if (add != 0)
            //     atomicAdd(htl + tid, add);

            for (int j = 16; j >= 1; j = j >> 1) {
                // Count_T t_min = __shfl_down_sync(0xffffffff, thread_min, j);
                Count_T t_max = __shfl_down_sync(0xffffffff, max_count, j);
                if (tid < j) {
                    // thread_min = min(thread_min, t_min);
                    max_count = max(max_count, t_max);
                }
            }

            max_count = __shfl_sync(0xffffffff, max_count, 0);
            if (add != 0 && max_count <= 128) {

                atomicAdd(htl + tid, add);
            }

            if (max_count > 128) {
                unsigned int old_id = 0;
                unsigned int new_id = 0;

                if (tid == 0) {
                    new_id = atomicAdd(mem_id, 1);
                    old_id = atomicMax(htl + 31, new_id);
                }
                new_id = __shfl_sync(0xffffffff, new_id, 0);
                old_id = __shfl_sync(0xffffffff, old_id, 0);

                if (old_id > new_id) {
                    if (tid == 0)
                        printf("path 1\n");
                    Count_T *hth = hash_table_high + old_id * 128;

                    if (tid == 0)
                        atomicAdd(hth + 124, 1);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        if (hash_mask[i * 4 + warp_mask_id] & warp_mask) {
                            atomicAdd(hth + i * WARP_SIZE + tid, 1);
                        }
                    }
                    __threadfence();
                    if (tid == 0)
                        atomicSub(hth + 124, 1);
                } else {
                    Count_T *hth = hash_table_high + new_id * 128;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        if (hash_mask[i * 4 + warp_mask_id] & warp_mask) {
                            atomicAdd(hth + i * WARP_SIZE + tid, 1);
                        }
                    }
                    if (old_id != 0) {
                        if (tid == 0) {
                            // printf("copy\n");
                            // printf("old_id: %u, new_id: %u\n", old_id, new_id);
                            // int try_count = 0;
                            // Count_T hth_val;
                            
                            volatile Count_T *hth_old_ready = hash_table_high + old_id * 128 + 124;
                            while (*hth_old_ready != 0) {}

                            // while (__ldcv(hash_table_high + old_id * 128 + 124) != 0) {}

                            // while ((hth_val = hash_table_high[old_id * 128 + 124]) != 0) {
                            // }


                            // while ((hth_val = hash_table_high[old_id * 128 + 124]) != 0 && try_count <= 1024) {
                            //     // printf("%u\n", hash_table_high[old_id * 128 + 124]);
                            //     try_count++;
                            // }
                            // printf("hth_val: %u\n", hth_val);
                            // for (int c = 0; c < 3; ++c)
                            // printf("old_id: %u, new_id: %u, hth_val: %u, hth_2_val: %u\n", old_id, new_id, hth_val);
                            // for (int c = 0; c < 10; ++c)
                            //     printf("%d:%u\t",c, hash_table_high[c * 128 + 124]);
                            // printf("\n");
                        }
                        
                        // if (tid == 0)
                        // volatile Count_T *hth_old = hash_table_high + old_id * 128;
                        for (int i = tid; i < 124; i += WARP_SIZE) {
                            // atomicAdd(hth + i, hash_table_high[old_id * 128 + i]);
                            // atomicAdd(hth + i, hth_old[i]);
                            // atomicAdd(hth + i, __ldcv(hash_table_high + old_id * 128 + i));
                        }

                    } 
                    __threadfence();
                    if (tid == 0)
                        atomicSub(hth + 124, 1);
                }

                
                


                // if (old_id != 0) {
                //     if (tid == 0) {
                //         printf("old_id: %u, new_id: %u\n", old_id, new_id);
                //         for (int c = 0; c < 10; ++c)
                //             printf("%d:%u\t",c, hash_table_high[c * 128 + 124]);
                //         printf("\n");
                //     }
                // }
                // if (old_id != 0)
                // {
                //     if (tid == 0) {
                //         // printf("%lu, %lu\n", hth + 124, hash_table_high + 2 * 128 + 124);
                //         printf("final: old_id: %u, new_id: %u, hth_124_val: %u\n", old_id, new_id, hth[124]);
                //         printf("t + 1: %u\n", );
                //     }
                // }
            } 
            

            
            

            
            // // printf("old_id: %u\n", old_id);
            // if (old_id != 0) {
            //     // printf("start\n");
            //     // if (old_id * 128 + 124 >= 1024 * 1024) {
            //     //     printf("err8\n");
            //     // }
            //     int try_count = 0;
            //     while (hash_table_high[old_id * 128 + 124] != 0 && try_count <= 1024) {
            //         // printf("%u\n", hash_table_high[old_id * 128 + 124]);
            //         try_count++;
            //     }
            //     // printf("old_id: %u, new_id: %u\n", old_id, new_id);
            //     // printf("val: %u\n", hash_table_high[old_id * 128 + 124]);
            //     // printf("move: %d\n", try_count);
            // }

            

            // if (max_count > 128) {
                
            // }




            // if (tid == 0) {
            //     new_id = 0;
            //     old_id = 0;
            //     // printf("%u\n", max_count);
            //     if (max_count > 128) {
            //         // upgrade
            //         new_id = atomicAdd(mem_id, 1);
            //         // atomicAdd(hash_table_high + new_id * 128 + 124, 1);
            //         // printf("new_id: %u\n", new_id);
            //         old_id = atomicMax(htl + 31, new_id);
            //         // printf("old_id: %u\n", old_id);
            //         if (old_id != 0) {
            //             // printf("start\n");
            //             // if (old_id * 128 + 124 >= 1024 * 1024) {
            //             //     printf("err8\n");
            //             // }
            //             int try_count = 0;
            //             while (hash_table_high[old_id * 128 + 124] != 0 && try_count <= 1024) {
            //                 // printf("%u\n", hash_table_high[old_id * 128 + 124]);
            //                 try_count++;
            //             }
            //             // printf("old_id: %u, new_id: %u\n", old_id, new_id);
            //             // printf("val: %u\n", hash_table_high[old_id * 128 + 124]);
            //             // printf("move: %d\n", try_count);
            //         }
            //     }
            // }
            // __threadfence_block();
            // if (old_id != 0) {
            //     if (tid == 0)
            //         printf("copy\n");
            //     for (int i = tid; i < 124; i += WARP_SIZE) {
            //         atomicAdd(hash_table_high + new_id * 128 + i, hash_table_high[old_id * 128 + i]);
            //     }
            // }

            // if (max_count > 128) {
            //     Count_T *hth = hash_table_high + new_id * 128;
            //     // if (tid != 0)
            //         atomicAdd(hth + 124, 1);
            //     #pragma unroll
            //     for (int i = 0; i < 4; ++i) {
            //         if (hash_mask[i * 4 + warp_mask_id] & warp_mask) {
            //             atomicAdd(hth + i * WARP_SIZE + tid, 1);
            //         }
            //     }
            //     atomicSub(hth + 124, 1);
            // }

            // if (tid == 0) {
            //     new_id = 0;
            //     old_id = 0;
            //     // printf("%u\n", max_count);
            //     if (max_count > 128) {
            //         // upgrade
            //         new_id = atomicAdd(mem_id, 1);
            //         // printf("new_id: %u\n", new_id);
            //         old_id = atomicMax(htl + 31, new_id);
            //         if (old_id != 0) {
            //             while (hash_table_high[old_id * 128 + 124] != 0) {}
            //         }
            //     }
            // }
            // __threadfence_block();
            // if (old_id != 0) {
            //     for (int i = tid; i < 128; i += WARP_SIZE) {
            //         atomicAdd(hash_table_high + new_id * 128 + i, hash_table_high[old_id * 128 + i]);
            //     }
            // }

            // if (max_count > 128) {
            //     Count_T *hth = hash_table_high + new_id;
            
            //     atomicAdd(hth + 124, 1);
            //     #pragma unroll
            //     for (int i = 0; i < 4; ++i) {
            //         if (hash_mask[i * 4 + warp_mask_id] & warp_mask) {
            //             atomicAdd(hth + i * WARP_SIZE + tid, 1);
            //         }
            //     }
            //     atomicSub(hth + 124, 1);
            // }

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
            // if (tid == 31 && thm != 0)
            //     printf("%u\n", thm & 0b00001111);
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

        if (htl[31] > BUFFER_START) {
            // insert_high();
            // printf("err3\n");
            Count_T *hth = hash_table_high + (htl[31] - BUFFER_START) * 128;
            
            // atomicAdd(hth + 124, 1);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (hash_mask[j * 4 + warp_mask_id] & warp_mask) {
                    // atomicAdd(hth + i * WARP_SIZE + tid, 1);
                    // if (i * WARP_SIZE + tid >= 124) {
                    //     printf("err6\n");
                    // }
                    thread_min_high = min(thread_min_high, hth[j * WARP_SIZE + tid]);
                    // if (thread_min_high < 102400) {
                        // printf("thread_min_high: %u\n", thread_min_high);
                    // } 
                }
            }
            // atomicSub(hth + 124, 1);
            // printf("thread_min_high: %u\n", thread_min_high);
            // thread_min_low += thread_min_high;

            for (int j = 16; j >= 1; j = j >> 1) {
                Count_T t_low = __shfl_down_sync(0xffffffff, thread_min_low, j);
                Count_T t_high = __shfl_down_sync(0xffffffff, thread_min_high, j);
                if (tid < j) {
                    thread_min_low = min(thread_min_low, t_low);
                    thread_min_high = min(thread_min_high, t_high);
                }
            }

            if (tid == 0) {
                count[i] = thread_min_low + thread_min_high;
            }

        } else {
            for (int j = 16; j >= 1; j = j >> 1) {
                Count_T t_low = __shfl_down_sync(0xffffffff, thread_min_low, j);
                if (tid < j) {
                    thread_min_low = min(thread_min_low, t_low);
                }
            }
            if (tid == 0) {
                count[i] = thread_min_low;
            }
        }
        // __threadfence_block();

        

        // if (tid == 0) {
            // if (thread_min_low > 102400) {
            //     printf("err4\n");
            // }

            // if (thread_min_high > 102400) {
            //     printf("err5\n");
            // }

            // if (thread_min_high != 0) {
            //     printf("thread_min_high: %u\n", thread_min_high);
            // }

            // count[i] = thread_min_low + thread_min_high;

            // if (thread_min_low > 4294967000u) {
            //     unsigned int *c = static_cast<unsigned int *> (debug);
            //     atomicAdd(c, 1);
            // }
        // }
    }
    // if (wid == 0 && tid == 0) {
    //     printf("kernel mem_id: %u\n", *mem_id);
    // }
}

// #define PRELOAD_SIZE 32

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
