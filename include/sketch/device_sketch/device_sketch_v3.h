#pragma once

namespace qsketch {

#ifdef QSKETCH_KERNEL_NAMESPACE
    #undef QSKETCH_KERNEL_NAMESPACE
#endif
#define QSKETCH_KERNEL_NAMESPACE Sketch_GPU_Sub_Warp_Level_Kernel

namespace QSKETCH_KERNEL_NAMESPACE {

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void calculation_s1_hash(
    size_t &hash_mask_id, Count_T **ht,
    
    Device_Hash_Table<Count_T> &dht,
    const size_t &p, 
    const Hash_Function &hash,
    Device_Seed<Seed_T> &ds,
    uchar *hash_mask_table,
    const size_t &hash_mask_table_sz,
    const size_t &hash_mask_sz,
    const Key_T &v,
    const size_t &tid,
    const size_t &sid,
    const size_t &stid,
    const uint &smask
    ) 
{

    auto &n = dht.n;
    auto &m = dht.m;
    auto hash_table = dht.table;

    auto &seed_sz = ds.seed_sz;
    auto seed = ds.seed;

    Hashed_T offset;
    if (tid % m == 0) {
        // Hashed_T hv = hash(seed, seed_sz, v) % p;
        Hashed_T hv = hash(seed, seed_sz, v);
        offset = (hv % n) * m;
        // offset = (hv % n) * 8;
        hash_mask_id = (hv % hash_mask_table_sz) * hash_mask_sz;
    }
    offset = __shfl_sync(smask, offset, sid * m);
    hash_mask_id = __shfl_sync(smask, hash_mask_id, sid * m);
    
    *ht = hash_table + offset;
}


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void insert_high (
    Count_T *hash_table, 
    // size_t m,
    uchar *hash_mask,   
    const size_t &tid,
    const uint &warp_mask_id, 
    const uchar &warp_mask)
{
    // return;
    if (hash_mask[warp_mask_id] & warp_mask)
        atomicAdd(hash_table + tid, 1);
}


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline uint atomic_malloc(
    Count_T *htl, 
    uint *next_level_id)
{
    uint id = 0;
    uint old = atomicCAS(htl, 0, 1);
    if (old == 0) {
        id = atomicAdd(next_level_id, 1);
        *htl = id;
    } else {
        while (id <= 1) {
            id = max(id, __ldcv(htl));
            __threadfence_block();
        }
    }
    return id;
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void calculation_s2_insert(
    
    const Count_T &cv,
    const Count_T &htl_next_level_id, // htl_7
    Device_Hash_Table<Count_T> &dht_low,
    Device_Hash_Table<Count_T> &dht_high,
    Count_T *htl,
    uchar *hash_mask,
    const size_t &hash_mask_sz,

    const uint &warp_mask_id,
    const uchar &warp_mask,

    // debug 
    const Key_T &v,
    const size_t &tid,
    const size_t &sid,
    const size_t &stid,
    const uint &smask
    ) 
{
    const auto &SUB_WARP_SIZE = dht_low.m;
    __syncwarp(0xffffffff);
    if ( __any_sync(0xffffffff, htl_next_level_id > default_values::DEVICE_NEXT_LEVEL_ID_START)) {
        #pragma unroll
        for (size_t i = 0; i < (default_values::DEVICE_WARP_SIZE); i += SUB_WARP_SIZE)
        {
            Count_T id = __shfl_sync(0xffffffff, htl_next_level_id, i);
            __syncwarp(0xffffffff);
            if (id > default_values::DEVICE_NEXT_LEVEL_ID_START) {
                uchar *hmh = hash_mask + (i / SUB_WARP_SIZE) * hash_mask_sz; // hmh: hash_mask_high
                Count_T *hth = dht_high.table + (id - default_values::DEVICE_NEXT_LEVEL_ID_START) * default_values::DEVICE_WARP_SIZE; // hth: hash_table
                // insert_high_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_table_high, id, hash_mask_high, tid, warp_mask_id, warp_mask, v);
                // printf("%s\n", );
                // hth = dht_high.table;
                // if (tid == 0)
                //     printf("%u\n", id);
                // printf("%lu\n", size_t((id - default_values::DEVICE_NEXT_LEVEL_ID_START) * default_values::DEVICE_WARP_SIZE));
                insert_high<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hth, hmh, tid, warp_mask_id, warp_mask);
            }
        }
    }
    __syncwarp(0xffffffff);
    unsigned int insert_low = __any_sync(smask, htl_next_level_id > default_values::DEVICE_NEXT_LEVEL_ID_START);
    Count_T max_count = 0;
    Count_T add = 0;

    if (insert_low == 0){
        uchar thm = hash_mask[tid/2];
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
    unsigned int update_low = __any_sync(smask, max_count > 128);
    unsigned int update_low_warp = __any_sync(0xffffffff, max_count > 128);

    if (add != 0 && (update_low == 0)) {
        // if (stid == 7) {
        //     uchar thm = hash_mask[tid/2];
        //     thm = thm >> ((tid % 2) * 4);
        //     printf("error: %u\n", uint(thm));
        // }
        atomicAdd(htl + stid, add);
    }
    
    if (update_low_warp) {
        unsigned int id = 0;
        if (tid % SUB_WARP_SIZE == 0 && update_low) {
            id = atomic_malloc<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(htl + SUB_WARP_SIZE - 1, dht_low.next_level_id);
        }
        __threadfence_block();

        #pragma unroll
        for (unsigned int idf = 0; idf < default_values::DEVICE_WARP_SIZE; idf += SUB_WARP_SIZE) {
            // unsigned int mask = __activemask();
            unsigned int idv = __shfl_sync(0xffffffff, id, idf);
            if (idv > default_values::DEVICE_NEXT_LEVEL_ID_START){
                uchar *hmh = hash_mask + (idf / SUB_WARP_SIZE) * hash_mask_sz; //hmh: hash_mask_high
                Count_T *hth = dht_high.table + (idv - default_values::DEVICE_NEXT_LEVEL_ID_START) * default_values::DEVICE_WARP_SIZE; // hth: hash_table
                // insert_high_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_table_high, idv, hash_mask_high, tid, warp_mask_id, warp_mask, v);
                insert_high<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hth, hmh, stid, warp_mask_id, warp_mask);
            }
        }
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_kernel(Key_T *keys, size_t sz,
    Device_Hash_Table<Count_T> dht_low,
    Device_Hash_Table<Count_T> dht_high,
    size_t p, 
    const Hash_Function &hash,
    Device_Seed<Seed_T> ds,
    uchar *hash_mask_table,
    size_t hash_mask_table_sz,
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    size_t tid = threadIdx.x % warpSize;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    if (e > sz) {
        e = sz;
    }
    
    // auto &n_low = dht_low.n;
    auto &m_low = dht_low.m;
    // auto hash_table_low = dht_low.table;

    // auto &seed_sz = ds.seed_sz;
    // auto seed = ds.seed;



    size_t sub_warp_num = warpSize / m_low;
    size_t sid = tid / m_low; // sub warp id
    size_t stid = tid % m_low; // sub warp thread id
    uint smask = ((1u << m_low) - 1) << (sid * m_low); // sub warp mask

    size_t hash_mask_sz = ceil<size_t>(m_low * sizeof(Count_T), device_bits<uchar>);
    const uint warp_mask_id = tid / bits<uchar>;
    const uchar warp_mask = 1u << ( tid % bits<uchar>);


    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     // printf("%lu\n", m_low);
    //     printf("hash_mask_table_sz: %lu\n", hash_mask_table_sz);
    //     printf("hash_mask_sz: %lu\n", hash_mask_sz);
    // }

    __shared__ uchar hash_mask[16];
    for (size_t i = b + sid; i < e; i += sub_warp_num) {

        Key_T v = keys[i]; // load 1
        
        Count_T *htl;
        size_t hash_mask_id;

        calculation_s1_hash<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_mask_id, &htl,
            dht_low,
            p,
            hash,
            ds,
            hash_mask_table,
            hash_mask_table_sz,
            hash_mask_sz,
            v,
            tid,
            sid,
            stid,
            smask
        );

        Count_T htl_next_level_id = htl[m_low - 1];
        Count_T cv = htl[stid];
        if (stid < hash_mask_sz) {
            hash_mask[sid * hash_mask_sz + stid] = hash_mask_table[hash_mask_id + stid];
        }
        __threadfence_block();

        calculation_s2_insert<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            cv,
            htl_next_level_id, // htl_7
            dht_low,
            dht_high,
            htl,
            hash_mask,
            hash_mask_sz,

            warp_mask_id,
            warp_mask,

            // debug 
            v,
            tid,
            sid,
            stid,
            smask
        );
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline Count_T search_high (
    const Count_T &Count_MAX,
    Count_T *hth, 
    uchar *hash_mask,   
    const size_t &tid,
    const uint &warp_mask_id, 
    const uchar &warp_mask
    )
{
    Count_T thread_min_high = Count_MAX;
    if (hash_mask[warp_mask_id] & warp_mask) {
        thread_min_high = hth[tid];
    }
    return thread_min_high;
}


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void calculation_s2_search(
    Count_T *count,
    const Count_T Count_MAX,
    const Count_T &cv,
    const Count_T &htl_next_level_id, // htl_7
    Device_Hash_Table<Count_T> &dht_low,
    Device_Hash_Table<Count_T> &dht_high,
    Count_T *htl,
    uchar *hash_mask,
    const size_t &hash_mask_sz,

    const uint &warp_mask_id,
    const uchar &warp_mask,

    // debug 
    const Key_T &v,
    const size_t &tid,
    const size_t &sid,
    const size_t &stid,
    const uint &smask
    ) 
{
    const auto &SUB_WARP_SIZE = dht_low.m;
    __shared__ Count_T thread_min_high[4];
    if (tid < 4) {
        thread_min_high[tid] = 0;
    }

    // Count_T thread_min_low = search_low<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(hash_mask, cv, tid);
    Count_T thread_min_low = Count_MAX;
    uchar thm = (hash_mask[tid/2]);
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

    __threadfence_block();
    __syncwarp();

    // printf("tid: %lu\n", tid);

    if (__any_sync(0xffffffff, htl_next_level_id > default_values::DEVICE_NEXT_LEVEL_ID_START)) {
        
        #pragma unroll
        for (size_t i = 0; i < (default_values::DEVICE_WARP_SIZE); i += SUB_WARP_SIZE)
        {           
            Count_T id = __shfl_sync(0xffffffff, htl_next_level_id, i);
            __syncwarp();
            // printf("tid: %lu, id: %u\n", tid, id);
            if (id > default_values::DEVICE_NEXT_LEVEL_ID_START) {
                uchar *hmh = hash_mask + (i / SUB_WARP_SIZE) * hash_mask_sz; // hmh: hash_mask_high
                Count_T *hth = dht_high.table + (id - default_values::DEVICE_NEXT_LEVEL_ID_START) * default_values::DEVICE_WARP_SIZE; // hth: hash_table
                Count_T tmhl = search_high<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(Count_MAX, hth, hmh, tid, warp_mask_id, warp_mask); // tmhl thread_min_high_local
                // printf("search_high\n");
                // printf("tid: %lu, tmhl: %u\n", tid, tmhl);
                // printf("tmhl: %u\n", tmhl);
                // Count_T thread_min_high_local_sub_warp = 
                //     search_high_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>
                //         (hash_mask_high, hash_table_high + (id - default_values::DEVICE_NEXT_LEVEL_ID_START) * default_values::DEVICE_WARP_SIZE, tid);
                for (int j = 16; j >= 1; j = j >> 1) {
                    Count_T t = __shfl_down_sync(0xffffffff, tmhl, j);
                    if (tid < j) {
                        tmhl = min(tmhl, t);
                    }
                }
                // printf("tid: %lu, tmhl: %u\n", tid, tmhl);
                if (tid == 0) {
                    // printf("tmh: %u\n", tmhl);
                    thread_min_high[i / SUB_WARP_SIZE] = tmhl;
                }
            }
        }
    }
    __threadfence_block();
    for (int j = 4; j >= 1; j = j >> 1) {
        Count_T t_low = __shfl_down_sync(0xffffffff, thread_min_low, j);
        if ((tid >= sid * SUB_WARP_SIZE) && (tid < sid * SUB_WARP_SIZE + j)) {
            thread_min_low = min(thread_min_low, t_low);
        }
    }
    // __threadfence_block();
    if (tid % SUB_WARP_SIZE == 0) {
        *count = thread_min_low + thread_min_high[sid];
        // printf("thread_min_low: %u, thread_min_high: %u, count: %u\n", thread_min_low, thread_min_high[sid], thread_min_low + thread_min_high[sid]);
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_kernel(Key_T *keys, size_t sz,
    Device_Hash_Table<Count_T> dht_low,
    Device_Hash_Table<Count_T> dht_high,
    size_t p, 
    const Hash_Function &hash,
    Device_Seed<Seed_T> ds,
    Count_T *count, Count_T Count_MAX,
    uchar *hash_mask_table,
    size_t hash_mask_table_sz,
    size_t work_load_per_warp,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    size_t tid = threadIdx.x % warpSize;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;
    
    if (e > sz) {
        e = sz;
    }

    // auto &n = dht.n;
    auto &m_low = dht_low.m;
    // auto hash_table = dht.table;

    // auto &seed_sz = ds.seed_sz;
    // auto seed = ds.seed;
    size_t sub_warp_num = warpSize / m_low;
    size_t sid = tid / m_low; // sub warp id
    size_t stid = tid % m_low; // sub warp thread id
    uint smask = ((1u << m_low) - 1) << (sid * m_low); // sub warp mask

    size_t hash_mask_sz = ceil<size_t>(m_low * sizeof(Count_T), device_bits<uchar>);
    const uint warp_mask_id = tid / bits<uchar>;
    const uchar warp_mask = 1u << ( tid % bits<uchar>);


    // if (wid == 0 && tid == 0) {
    //     printf("e: %lu\n", e);
    // }
    // if (blockIdx.x == 0) {
    //     e = 4;
    // } else {
    //     e = 0;
    // }

    __shared__ uchar hash_mask[16];
    for (size_t i = b + sid; i < e; i += sub_warp_num) {

        Key_T v = keys[i]; // load 1
        // Key_T v = keys[0];
        // printf("i: %lu, key: %u\n",i, v);

        Count_T *htl;
        size_t hash_mask_id;

        calculation_s1_hash<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_mask_id, &htl,
            dht_low,
            p,
            hash,
            ds,
            hash_mask_table,
            hash_mask_table_sz,
            hash_mask_sz,
            v,
            tid,
            sid,
            stid,
            smask
        );

        Count_T htl_next_level_id = htl[m_low - 1];
        Count_T cv = htl[stid];
        if (stid < hash_mask_sz) {
            hash_mask[sid * hash_mask_sz + stid] = hash_mask_table[hash_mask_id + stid];
        }
        __syncwarp();
        // printf("i: %lu\n", i);
        calculation_s2_search<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            count + i,
            Count_MAX,
            cv,
            htl_next_level_id, // htl_7
            dht_low,
            dht_high,
            htl,
            hash_mask,
            hash_mask_sz,

            warp_mask_id,
            warp_mask,

            // debug 
            v,
            tid,
            sid,
            stid,
            smask
            // ht,
            // m,
            // hash_mask,
            // tid,
            // sid,
            // stid,
            // warp_mask_id,
            // warp_mask
        );
    }
    
}

}
template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, 
    typename Hash_Function> // 
struct Sketch_GPU_Sub_Warp_Level : public Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> {
    using Base_Class = Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>;
    
    using Base_Class::gridDim;
    using Base_Class::blockDim;
    using Base_Class::nwarps;
    using Base_Class::nthreads;

    using Base_Class::prime_number;

    using Base_Class::dht_low;
    using Base_Class::dht_high;
    using Base_Class::ds;

    using Base_Class::hash_mask_table;
    size_t hash_mask_table_sz;
    // using Base_Class::hash_mask_table_sz;

    Sketch_GPU_Sub_Warp_Level(size_t _n_low, size_t _m_low,
            size_t _n_high, size_t _m_high, 
            size_t _ss = default_values::seed_sz, size_t _n_hash_mask_table = default_values::HASH_MASK_TABLE_SZ) : 
        Base_Class(_n_low, _m_low, _n_high, _m_high), hash_mask_table_sz(_n_hash_mask_table)
    {        
#ifdef QSKETCH_DEBUG
        std::cout << "Sketch_GPU_Sub_Warp_Level" << (_m_low) << std::endl;
        // if (_m >= default_values::DEVICE_WARP_SIZE) {
        //     std::cerr << "error : _m must be greater than WARP_SIZE" << std::endl;
        // }
#endif

        ds.resize(1, _ss);

        hash_mask_table = gpu_tool<uchar>.zero(hash_mask_table, _n_hash_mask_table * _m_low * sizeof(Count_T));
        uchar *hash_mask_table_host = generate_hashmask(nullptr, _n_hash_mask_table, _m_low * sizeof(Count_T), default_values::HASH_MASK_ONES, sizeof(uint));
        CUDA_CALL(cudaMemcpy(hash_mask_table, hash_mask_table_host, sizeof(uchar) * _n_hash_mask_table * _m_low * sizeof(Count_T),
            cudaMemcpyHostToDevice));
        delete hash_mask_table_host;
    }

    // virtual size_t number_of_buckets(size_t insert_keys_sz, size_t m, double factor) {
    //     size_t table_sz = ((insert_keys_sz / factor) * default_values::HASH_MASK_ONES) / m;
    //     return table_sz;
    // }

    virtual int insert(Key_T *keys, size_t keys_sz) {
        using namespace QSKETCH_KERNEL_NAMESPACE;
        size_t work_load_per_warp = ceil<size_t>(keys_sz, Base_Class::nwarps);
        // std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;
#ifdef QSKETCH_ASYNC_MEMCPY

#endif
        CUDA_CALL((insert_kernel<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>
            <<<Base_Class::gridDim, Base_Class::blockDim>>>(
                keys, keys_sz,
                // Base_Class::table, Base_Class::n, Base_Class::m, Base_Class::prime_number,
                dht_low,
                dht_high,
                prime_number,
                Hash_Function(),
                // Base_Class::seed, Base_Class::seed_sz,
                ds,
                hash_mask_table,
                hash_mask_table_sz,
                work_load_per_warp,
                nullptr
            )));
        cudaDeviceSynchronize();
        // std::cout << "insert end" << std::endl;
        #ifdef QSKETCH_DEBUG
            dht_low.print_next_level_id();
        #endif 

        return 0;
    }
    virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
        using namespace QSKETCH_KERNEL_NAMESPACE;
        size_t work_load_per_warp = ceil<size_t>(keys_sz, Base_Class::nwarps);
        // std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;
        CUDA_CALL((search_kernel<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>
            <<<Base_Class::gridDim, Base_Class::blockDim>>>(
                keys, keys_sz,
                // Base_Class::table, Base_Class::n, Base_Class::m, Base_Class::prime_number,
                dht_low,
                dht_high,
                prime_number,
                Hash_Function(),
                // Base_Class::seed, Base_Class::seed_sz,
                ds,
                count, std::numeric_limits<Count_T>::max(),
                hash_mask_table,
                hash_mask_table_sz,
                work_load_per_warp,
                nullptr
            )));
        cudaDeviceSynchronize();

        // thrust::device_vector<Count_T> dvc(dht_low.table, dht_low.table + dht_low.table_total_sz);
        // auto avc = average_diff(dvc);
        // std::cout << "average count: " << avc << std::endl;

        // thrust::host_vector<Count_T> hvc(dvc);
        // for (auto &val : hvc) {
        //     std::cout << val << std::endl;
        // }

        return 0;
    }

};

}