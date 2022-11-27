#pragma once

namespace qsketch {

#ifdef QSKETCH_KERNEL_NAMESPACE
    #undef QSKETCH_KERNEL_NAMESPACE
#endif
#define QSKETCH_KERNEL_NAMESPACE Sketch_GPU_SF_Host_Sub_Warp_Kernel

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
        // printf("key: %u, id :%u search\n", v, offset);

        hash_mask_id = (hv % hash_mask_table_sz) * hash_mask_sz;
        // printf("key: %u, id :%lu search\n", v, (hv % hash_mask_table_sz));
        // printf("tid: %lu, offset: %u\n", size_t(tid), offset);
    }
    offset = __shfl_sync(smask, offset, sid * m);
    hash_mask_id = __shfl_sync(smask, hash_mask_id, sid * m);
    
    *ht = hash_table + offset;
}

template <typename Count_T>
__global__ void update(Count_T *ptr, Count_T val) {
    atomicMax(ptr, val);
}

template <typename Count_T>
__global__ void update_table(Count_T *it, size_t sz, Count_T *st, uint times) {
    // size_t work_load_per_block = sz / times / gridDim.x;
    size_t st_sz = sz / times;
    size_t work_load_per_block = ceil<size_t>(st_sz, gridDim.x);
    size_t b = blockIdx.x * work_load_per_block;
    size_t e = (blockIdx.x + 1) * work_load_per_block;
    if (e > st_sz) {
        e = st_sz;
    }
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("work_load_per_block: %lu\n", work_load_per_block);
    // }
    // if (threadIdx.x == 0) {
    //     printf("e: %lu\n", e);
    // }
    for (size_t i = b + threadIdx.x; i < e; i += blockDim.x) {
        // printf("i: %lu\n", i);
        Count_T iniv = 0;
        for (uint j = 0; j < times; ++j) {
            Count_T itv = it[i * times + j];
            // Count_T itv = it[1];
            iniv = max(iniv, itv);
        }
        st[i] = iniv;
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
    const uchar &warp_mask,
    const Key_T &v
    )
{
    Count_T thread_min = Count_MAX;
    if (hash_mask[warp_mask_id] & warp_mask) {
        thread_min = min(thread_min, hash_table[stid]);
        // printf("key: %u, stid :%lu search\n", v, stid);
    }

    for (int j = m / 2; j >= 1; j = j >> 1) {
        Count_T t = __shfl_down_sync(0xffffffff, thread_min, j);
        if ((tid >= sid * m) && (tid < sid * m + j)) {
            thread_min = min(thread_min, t);
        }
    }
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
            warp_mask,
            v
        );
    }
    
}

}

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, 
    typename Hash_Function> // 
struct Sketch_GPU_SF_Host_Sub_Warp : public Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> {
    using Base_Class = Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>;
    
    using Base_Class::gridDim;
    using Base_Class::blockDim;
    using Base_Class::nwarps;
    using Base_Class::nthreads;

    using Base_Class::prime_number;

    using Base_Class::dht_low;
    using Base_Class::dht_high;

    // Device_Hash_Table<Count_T> &dht_insert = dht_high;
    Device_Hash_Table<Count_T> &dht_search = dht_low;
    // Zero_Copy_Hash_Table<Count_T> dht_insert;
    // Host_Hash_Table<std::atomic<Count_T>> dht_insert;
    Atomic_Hash_Table<Count_T> dht_insert;

    using Base_Class::ds;
    Host_Seed<Seed_T> hs;

    using Base_Class::hash_mask_table;
    size_t hash_mask_table_sz;
    // using Base_Class::hash_mask_table_sz;

    uint *index_hash_mask_table = nullptr;
    uint *index_hash_mask_table_host = nullptr;


    // uchar * F_hash_mask_table = nullptr;
    // size_t F_hash_mask_table_sz;

    uint times;

    Sketch_GPU_SF_Host_Sub_Warp(size_t _n, size_t _m, uint _times,
            size_t _ss = default_values::seed_sz, size_t _n_hash_mask_table = default_values::HASH_MASK_TABLE_SZ) : 
        // Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(_n, _m) 
        // Base_Class(_n, _m, _ss + default_values::HASH_MASK_ONES, _n_hash_mask_table)
        Base_Class(_n, _m), hash_mask_table_sz(_n_hash_mask_table), times(_times)
    {        

        // printf("dht_search.m: %lu\n", dht_search.m);

#ifdef QSKETCH_DEBUG
        std::cout << "Sketch_GPU_SF_Host_Sub_Warp" << (_m) << std::endl;
        // if (_m >= default_values::WARP_SIZE) {
        //     std::cerr << "error : _m must be greater than WARP_SIZE" << std::endl;
        // }
#endif

        // size_t hash_mask_sz = ceil<size_t>(_m, device_bits<uchar>);
        // F_hash_mask_table = resize_hash_mask_table(hash_mask_table, hash_mask_table_sz * hash_mask_sz, nullptr, times);
        // F_hash_mask_table_sz = 

#ifdef QSKETCH_DEBUG
        if (times < 1) {
            std::cerr << "error : times should be greater than 0" << std::endl; // for now
        }
#endif        
        dht_insert.resize(_n, _m * times);
        // std::cout << "resize end" << std::endl;

        ds.resize(1 + default_values::HASH_MASK_ONES, _ss); // each thread only need one seed
        // hs = ds;
        // hs.copy_from_device<Device_Seed<Seed_T>>(ds);
        copy_seed_from_device();
        hash_mask_table = gpu_tool<uchar>.zero(hash_mask_table, hash_mask_table_sz * _m);
        // index_hash_mask_table = gpu_tool<uint>.zero(index_hash_mask_table, hash_mask_table_sz * default_values::HASH_MASK_ONES);

        uchar *hash_mask_table_host = generate_hashmask(nullptr, hash_mask_table_sz, _m,
            default_values::HASH_MASK_ONES, 0, &index_hash_mask_table_host);

        CUDA_CALL(cudaMemcpy(hash_mask_table, hash_mask_table_host, sizeof(uchar) * _n_hash_mask_table * _m,
            cudaMemcpyHostToDevice));
        // CUDA_CALL(cudaMemcpy(index_hash_mask_table, index_hash_mask_table_host, 
        //     sizeof(uint) * hash_mask_table_sz * default_values::HASH_MASK_ONES, cudaMemcpyHostToDevice));

        // for (int i = 0; i < hash_mask_table_sz; ++i) {
        //     for (int j = 0; j < default_values::HASH_MASK_ONES; ++j) {
        //         std::cout << index_hash_mask_table_host[i * default_values::HASH_MASK_ONES + j] << ",";
        //     }
        //     std::cout << std::endl;
        // }

        delete []hash_mask_table_host;
        // delete []index_hash_mask_table_host;
    }

    void copy_seed_from_device()
    {
        hs.seed_num = ds.seed_num;
        hs.seed_sz = ds.seed_sz;
        hs.seed_total_sz = ds.seed_total_sz;
        if (hs.seed) {
            delete [] hs.seed;
        }
        hs.seed = cpu_tool<Seed_T>.zero(nullptr, hs.seed_total_sz);
        CUDA_CALL(cudaMemcpy(hs.seed, ds.seed, sizeof(Seed_T) * hs.seed_total_sz, cudaMemcpyDeviceToHost));
        // return *this;
    }

    void update() {
        using namespace QSKETCH_KERNEL_NAMESPACE;
        const uint batch_buckets = 1 << 16;
        const uint num_counts = batch_buckets * dht_insert.m;
        Count_T *device_buffer = nullptr;
        device_buffer = gpu_tool<Count_T>.zero(device_buffer, num_counts);

        size_t bs = dht_insert.n;
        auto it = dht_insert.table;
        Count_T *st = dht_search.table;
        while (bs != 0) {
            size_t cpd = std::min<size_t>(bs, batch_buckets);
            CUDA_CALL(cudaMemcpy(device_buffer, it, sizeof(Count_T) * cpd * dht_insert.m, cudaMemcpyHostToDevice));
            cudaDeviceSynchronize();
            // std::cout << cpd * dht_insert.m << "," << dht_insert.m << "," << times << std::endl;
            CUDA_CALL((update_table<Count_T><<<1024, 32>>>(device_buffer, cpd * dht_insert.m, st, times)));
            
            cudaDeviceSynchronize();
            it += cpd * dht_insert.m;
            st += cpd * dht_search.m;
            bs -= cpd;
        }
        gpu_tool<Count_T>.free(device_buffer);
    }

    virtual int insert(Key_T *keys, size_t keys_sz) {
        using namespace QSKETCH_KERNEL_NAMESPACE;


        auto &n = dht_search.n;
        auto &seed_sz = hs.seed_sz;
        auto seed = hs.seed;
        // auto &seed_sz = ds.seed_sz;
        // auto seed = ds.seed;
        // std::cout << "seed_sz: " << seed_sz << std::endl;
        auto &index_hash_mask_sz = default_values::HASH_MASK_ONES;


        Hash_Function hash;


        const size_t host_buffer_sz = 1 << 24;
        // const size_t device_buffer_sz = 1 << 24;
        // static Key_T host_keys[host_buffer_sz];
        Key_T *host_keys = new Key_T[host_buffer_sz];
        while (keys_sz != 0) {
            size_t cpd = std::min(keys_sz, host_buffer_sz);
            CUDA_CALL(cudaMemcpy(host_keys, keys, sizeof(Key_T) * cpd, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            // #pragma omp parallel for static
            for (size_t i = 0; i < cpd; ++i) {
                // std::cout << "p0:" << i << std::endl;
                Key_T v = host_keys[i];
                // std::cout << "p1:" << i << std::endl;

                Hashed_T hv = hash.host_hash(seed, seed_sz, v);
                Hashed_T id = hv % n;
                                // std::cout << "p2:" << i << std::endl;

                std::atomic<Count_T> *insert_bucket = dht_insert.table + id * dht_insert.m;
                // Count_T *search_bucket = dht_search.table + id * dht_search.m;
                uint *index_hash_mask = index_hash_mask_table_host + (hv % hash_mask_table_sz) * index_hash_mask_sz;
                // std::cout << "p3:" << i << std::endl;

                for (uint j = 0; j < default_values::HASH_MASK_ONES; ++j) {
                    Hashed_T hv = hash.host_hash(seed + j * seed_sz, seed_sz, v);
                    Hashed_T id = hv % times;
                    // Count_T old = atomicAdd(insert_bucket + index_hash_mask[j] * times + id, 1);
                    // atomicMax(search_bucket + index_hash_mask[j], old + 1);
                                    // std::cout << "p4:" << j << std::endl;
                    // std::cout << "iid: " << index_hash_mask[j] << "," << times << "," << id << std::endl;
                    Count_T old = insert_bucket[index_hash_mask[j] * times + id].fetch_add(1);
                    // update<Count_T><<<1, 1>>>(search_bucket + index_hash_mask[j], old + 1);

                }

            }

            keys += cpd;
            keys_sz -= cpd;
        }
        delete []host_keys;
        // update
        update();
        // std::cout << "dht_search.table_total_sz: " << dht_search.table_total_sz << std::endl;
        // cudaDeviceSynchronize();

        


//         size_t work_load_per_warp = ceil<size_t>(keys_sz, Base_Class::nwarps);
//         // std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;
// #ifdef QSKETCH_ASYNC_MEMCPY

// #endif
//         CUDA_CALL((insert_kernel<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>
//             <<<Base_Class::gridDim, Base_Class::blockDim>>>(
//                 keys, keys_sz,
//                 // Base_Class::table, Base_Class::n, Base_Class::m, Base_Class::prime_number,
//                 dht_search,
//                 dht_insert,
//                 index_hash_mask_table,
//                 default_values::HASH_MASK_ONES, // index_sz,
//                 hash_mask_table_sz,
//                 times,
//                 prime_number,
//                 Hash_Function(),
//                 // Base_Class::seed, Base_Class::seed_sz,
//                 ds,
//                 // hash_mask_table,
//                 // hash_mask_table_sz,
//                 work_load_per_warp,
//                 nullptr
//             )));
//         cudaDeviceSynchronize();
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

    virtual void clear() {
        dht_search.clear();
        dht_insert.clear();
    }

    // virtual size_t number_of_buckets(size_t insert_keys_sz, size_t m, double factor) {
    //     return 0;
    // }

    virtual ~Sketch_GPU_SF_Host_Sub_Warp() {
        // cudaFree(table);
        // cudaFree(seed);
        dht_search.free();
        dht_insert.free();
        ds.free();

        cudaFree(index_hash_mask_table);
        delete []index_hash_mask_table_host;
    }

};

}