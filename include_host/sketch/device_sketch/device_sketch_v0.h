#pragma once

namespace qsketch {

#ifdef QSKETCH_KERNEL_NAMESPACE
    #undef QSKETCH_KERNEL_NAMESPACE
#endif
#define QSKETCH_KERNEL_NAMESPACE Sketch_GPU_Thread_Kernel

namespace QSKETCH_KERNEL_NAMESPACE {

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, typename Hash_Function>
__global__ void insert_kernel(Key_T *keys, size_t sz,
    // Count_T *hash_table, size_t n, size_t m, 
    Device_Hash_Table<Count_T> dht,
    size_t p, 
    const Hash_Function &hash,
    // Seed_T *seed, size_t s_sz, 
    Device_Seed<Seed_T> ds,
    size_t work_load_per_warp,
    void *debug)
{


    /*
        p is a prime number, and p >> n_low * WARP_SIZE
    */

    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    size_t tid = threadIdx.x % warpSize;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;
    
    auto &n = dht.n;
    auto &m = dht.m;
    auto hash_table = dht.table;

    auto &seed_sz = ds.seed_sz;
    auto seed = ds.seed;

    if (e > sz) {
        e = sz;
    }

    for (size_t i = b + tid; i < e; i += warpSize) {
        Key_T v = keys[i];
        for (size_t j = 0; j < m; ++j) {
            Hashed_T hv = hash(seed + j, seed_sz, v);
            // Hashed_T hv = v;
            // size_t iv = (hv % p) % (n);
            size_t iv = (hv) % (n);
            atomicAdd(hash_table + j * n + iv, 1);
        }
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
    size_t work_load_per_warp,
    void *debug)
{

    /*
        p is a prime number, and p >> n_low * WARP_SIZE
    */

    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    size_t tid = threadIdx.x % warpSize;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    auto &n = dht.n;
    auto &m = dht.m;
    auto hash_table = dht.table;

    auto &seed_sz = ds.seed_sz;
    auto seed = ds.seed;
    
    if (e > sz) {
        e = sz;
    }

    for (size_t i = b + tid; i < e; i += warpSize) {
        Key_T v = keys[i];
        Count_T c = Count_MAX;
        for (size_t j = 0; j < m; ++j) {
            Hashed_T hv = hash(seed + j, seed_sz, v);
            // size_t iv = (hv % p) % (n);
            size_t iv = (hv) % (n);
            c = min(c, hash_table[j * n + iv]); 
        }
        count[i] = c;
    }   
}

}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, 
    typename Hash_Function> // 
struct Sketch_GPU_Thread : public Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> {
    using Base_Class = Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>;
    
    using Base_Class::gridDim;
    using Base_Class::blockDim;
    using Base_Class::nwarps;
    using Base_Class::nthreads;

    using Base_Class::prime_number;

    using Base_Class::dht_low;
    using Base_Class::dht_high;
    using Base_Class::ds;

    Sketch_GPU_Thread(size_t _n, size_t _m, 
            size_t _ss = default_values::seed_sz) : 
        // Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(_n, _m) 
        Base_Class(_n, _m)
    {        
#ifdef QSKETCH_DEBUG
        std::cout << "Sketch_GPU_Thread" << std::endl;
#endif

        ds.resize(_m, _ss); // each thread only need one seed
    }

    static size_t number_of_buckets(size_t insert_keys_sz, size_t m, double factor) {
        size_t table_sz_row = insert_keys_sz / factor;
        size_t table_sz_prime = find_greatest_prime(table_sz_row);
        // std::cout << "table_sz: " << table_sz_row << ", " << table_sz_prime  << std::endl;
        return table_sz_prime;
    }

    virtual int insert(Key_T *keys, size_t keys_sz) {
        using namespace QSKETCH_KERNEL_NAMESPACE;
        size_t work_load_per_warp = ceil<size_t>(keys_sz, Base_Class::nwarps);

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
                work_load_per_warp,
                nullptr
            )));
        cudaDeviceSynchronize();
        return 0;
    }
    virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
        using namespace QSKETCH_KERNEL_NAMESPACE;
        size_t work_load_per_warp = ceil<size_t>(keys_sz, Base_Class::nwarps);
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