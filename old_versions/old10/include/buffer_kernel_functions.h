#pragma once

// template <typename Key_T, typename Count_T>
// inline void insert_warp_mask_threadfence(const Key_T key, const Count_T count, Count_T *hash_table, unsigned int *hash_mask) {

// }


// inline void insert()
// inline void remove()
// inline void search()

// inline void increase()

template <typename T>
__device__ bool try_push(const T &val, T *buffer, size_t sz, 
    unsigned int *buffer_col_sz, unsigned int *bid, unsigned int *eid) {
    unsigned int id = atomicAdd(buffer_col_sz, 1);
    if (id <= sz) {
        buffer[id] = val;
        atomicAdd(eid, 1);
        return true;
    } else {
        atomicSub(buffer_col_sz, 1);
        return false;
    }
}

template <typename T>
__device__ bool try_pop(const T &val, T *buffer, size_t sz, 
    unsigned int *buffer_col_sz, unsigned int *bid, unsigned int *eid) {
    unsigned int id = atomicSub(buffer_col_sz, 1);
    if (id > 0) {
        // buffer[id] = val;
        val = buffer[id];
        atomicAdd(bid, 1);
        return true;
    } else {
        atomicAdd(buffer_col_sz, 1);
        return false;
    }
}



void insert_buffer(Key_T *keys, size_t sz,
    Key_T *buffer, size_t nc, size_t nr,
    unsigned int *buffer_col_sz, unsigned int *bid, unsigned int *eid,
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
        Hashed_T hv = hash(seed + tid * s_sz, s_sz, v) % nc;
        
        if (try_push(v, buffer + hv * nr, nr, buffer_col_sz + hv, bid + hv, eid + hv)) {
            i += 
        }
    }
}