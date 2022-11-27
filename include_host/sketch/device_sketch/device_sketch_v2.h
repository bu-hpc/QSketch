#pragma once

namespace qsketch {

#ifdef QSKETCH_KERNEL_NAMESPACE
    #undef QSKETCH_KERNEL_NAMESPACE
#endif
#define QSKETCH_KERNEL_NAMESPACE Sketch_GPU_Sub_Warp_Kernel

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
    ) {

    auto &n = dht.n;
    auto &m = dht.m;
    auto hash_table = dht.table;

    auto &seed_sz = ds.seed_sz;
    auto seed = ds.seed;

    Hashed_T offset;
    if (tid % m == 0) {
        // Hashed_T hv = hash(seed, seed_sz, v) % p;
        Hashed_T hv = hash(seed, seed_sz, v);// % p;
        offset = (hv % n) * m;
        hash_mask_id = (hv % hash_mask_table_sz) * hash_mask_sz;

        // printf("tid: %lu, offset: %u\n", size_t(tid), offset);
    }
    offset = __shfl_sync(smask, offset, sid * m);
    hash_mask_id = __shfl_sync(smask, hash_mask_id, sid * m);
    
    *ht = hash_table + offset;
}


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void calculation_s2_insert (
    Count_T *hash_table, 
    size_t m,
    uchar *hash_mask,   
    const size_t &stid,
    const uint &warp_mask_id, 
    const uchar &warp_mask)
{
    if (hash_mask[warp_mask_id] & warp_mask)
        atomicAdd(hash_table + stid, 1);
}


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_kernel(Key_T *keys, size_t sz,
    Device_Hash_Table<Count_T> dht,
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
    
    // auto &n = dht.n;
    auto &m = dht.m;
    // auto hash_table = dht.table;

    // auto &seed_sz = ds.seed_sz;
    // auto seed = ds.seed;

    // printf("wid: %lu, tid: %lu\n", wid, size_t(tid));

    size_t sub_warp_num = warpSize / m;
    size_t sid = tid / m;
    size_t stid = tid % m;
    uint smask = ((1u << m) - 1) << (sid * m);

    size_t hash_mask_sz = ceil<size_t>(m, device_bits<uchar>);
    const uint warp_mask_id = tid / bits<uchar>;
    const uchar warp_mask = 1u << ( tid % bits<uchar>);


    // if (wid == 0) 
    // {
    //     printf("sub_warp_num: %lu\n", sub_warp_num);
    //     // printf("hmts: %lu, hms: %lu\n", hash_mask_table_sz, hash_mask_sz);
    //     // printf("hello \n");
    //     // printf("m: %lu\n", m);
    // }

    // return ;

    __shared__ uchar hash_mask[16];
    for (size_t i = b + sid; i < e; i += sub_warp_num) {

        Key_T v = keys[i]; // load 1
        
        Count_T *ht;
        size_t hash_mask_id;

        calculation_s1_hash<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_mask_id, &ht,
            dht,
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

        if (stid < hash_mask_sz) {
            // printf("hm: %lu, %lu\n", sid * hash_mask_sz + stid, hash_mask_id + stid);
            hash_mask[sid * hash_mask_sz + stid] = hash_mask_table[hash_mask_id + stid];
        }
        calculation_s2_insert<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            ht,
            m,
            hash_mask,
            stid,
            warp_mask_id,
            warp_mask
        );
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__device__ inline void calculation_s2_search (
    Count_T *count,
    const Count_T &Count_MAX,
    Count_T *hash_table, 
    size_t m,
    uchar *hash_mask,   
    const size_t &tid,
    const size_t &sid,
    const size_t &stid,
    const uint &warp_mask_id, 
    const uchar &warp_mask
    )
{
    Count_T thread_min = Count_MAX;
    if (hash_mask[warp_mask_id] & warp_mask)
        thread_min = min(thread_min, hash_table[stid]);

    // for (int j = 16; j >= 1; j = j >> 1) {
    //     Count_T t = __shfl_down_sync(0xffffffff, thread_min, j);
    //     if (tid < j) {
    //         thread_min = min(thread_min, t);
    //     }
    // }
    // if (tid == 0) {
    //     *count = thread_min;
    // }
    for (int j = m / 2; j >= 1; j = j >> 1) {
        Count_T t = __shfl_down_sync(0xffffffff, thread_min, j);
        if ((tid >= sid * m) && (tid < sid * m + j)) {
            thread_min = min(thread_min, t);
        }
    }
    // __threadfence_block();
    if (stid == 0) {
        *count = thread_min;
    }
}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void search_kernel(Key_T *keys, size_t sz,
    // Count_T *hash_table, size_t n, size_t m, 
    Device_Hash_Table<Count_T> dht,
    size_t p, 
    const Hash_Function &hash,
    // Seed_T *seed, size_t s_sz, 
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
    auto &m = dht.m;
    // auto hash_table = dht.table;

    // auto &seed_sz = ds.seed_sz;
    // auto seed = ds.seed;
    size_t sub_warp_num = warpSize / m;
    size_t sid = tid / m;
    size_t stid = tid % m;
    uint smask = ((1u << m) - 1) << (sid * m);

    size_t hash_mask_sz = ceil<size_t>(m, device_bits<uchar>);
    const uint warp_mask_id = tid / bits<uchar>;
    const uchar warp_mask = 1u << ( tid % bits<uchar>);

    // if (wid == 0 && tid == 0) 
    // {
    //     printf("sub_warp_num: %lu\n", sub_warp_num);
    //     printf("m : %lu\n", m);
    //     // printf("hmts: %lu, hms: %lu\n", hash_mask_table_sz, hash_mask_sz);
    //     // printf("hello \n");
    //     // printf("m: %lu\n", m);
    // }

    __shared__ uchar hash_mask[16];
    for (size_t i = b + sid; i < e; i += sub_warp_num) {

        Key_T v = keys[i]; // load 1
        
        Count_T *ht;
        size_t hash_mask_id;

        calculation_s1_hash<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_mask_id, &ht,
            dht,
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

        if (stid < hash_mask_sz) {
            hash_mask[sid * hash_mask_sz + stid] = hash_mask_table[hash_mask_id + stid];
        }
        calculation_s2_search<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            count + i,
            // count,
            Count_MAX,
            ht,
            m,
            hash_mask,
            tid,
            sid,
            stid,
            warp_mask_id,
            warp_mask
        );
    }
    
}

}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, 
    typename Hash_Function> // 
struct Sketch_GPU_Sub_Warp : public Sketch_GPU_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> {
    using Base_Class = Sketch_GPU_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>;
    
    using Base_Class::gridDim;
    using Base_Class::blockDim;
    using Base_Class::nwarps;
    using Base_Class::nthreads;

    using Base_Class::prime_number;

    using Base_Class::dht_low;
    using Base_Class::dht_high;
    using Base_Class::ds;

    using Base_Class::hash_mask_table;
    // size_t hash_mask_table_sz;
    using Base_Class::hash_mask_table_sz;

    Sketch_GPU_Sub_Warp(size_t _n, size_t _m, 
            size_t _ss = default_values::seed_sz, size_t _n_hash_mask_table = default_values::HASH_MASK_TABLE_SZ) : 
        // Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(_n, _m) 
        Base_Class(_n, _m, _ss, _n_hash_mask_table)
    {        
#ifdef QSKETCH_DEBUG
        std::cout << "Sketch_GPU_Sub_Warp_" << (_m) << std::endl;
        // if (_m >= default_values::WARP_SIZE) {
        //     std::cerr << "error : _m must be greater than WARP_SIZE" << std::endl;
        // }
#endif

        // ds.resize(1, _ss); // each thread only need one seed

        // hash_mask_table = gpu_tool<uchar>.zero(hash_mask_table, _n_hash_mask_table * _m);
        // uchar *hash_mask_table_host = generate_hashmask(nullptr, _n_hash_mask_table, _m);
        // CUDA_CALL(cudaMemcpy(hash_mask_table, hash_mask_table_host, sizeof(uchar) * _n_hash_mask_table * _m,
        //     cudaMemcpyHostToDevice));

    }

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