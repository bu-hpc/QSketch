#pragma once

namespace qsketch {

#ifdef QSKETCH_KERNEL_NAMESPACE
    #undef QSKETCH_KERNEL_NAMESPACE
#endif
#define QSKETCH_KERNEL_NAMESPACE Sketch_GPU_SF_Host_Sub_Warp_Fly_Kernel

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
    Device_Hash_Table<Count_T> dht_insert,
    Device_Hash_Table<Count_T> dht_search,
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
    auto &m = dht_insert.m;
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

    __shared__ uchar hash_mask[16 * 8]; // times == 8;
    for (size_t i = b + sid; i < e; i += sub_warp_num) {

        Key_T v = keys[i]; // load 1
        
        Count_T *ht;
        size_t hash_mask_id;

        calculation_s1_hash<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>(
            hash_mask_id, &ht,
            dht_insert,
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
struct Sketch_GPU_SF_Host_Sub_Warp_Fly : public Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> {
    using Base_Class = Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>;
    
    using Base_Class::gridDim;
    using Base_Class::blockDim;
    using Base_Class::nwarps;
    using Base_Class::nthreads;

    using Base_Class::prime_number;

    using Base_Class::dht_low;
    using Base_Class::dht_high;

    Device_Hash_Table<Count_T> &dht_insert = dht_low;
    Device_Hash_Table<Count_T> &dht_search = dht_high;
    Host_Hash_Table<Count_T> hht_insert;
    Host_Hash_Table<Count_T> hht_search;

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
    size_t device_buffer_sz = 0; // it stores the tables of dht_insert and dht_search


    enum Mode
    {
        INSERT,
        SEARCH
    };
    Mode mode = Mode::INSERT;

    Sketch_GPU_SF_Host_Sub_Warp_Fly(size_t _n, size_t _m, uint _times, 
        uint device_buffer_sz = default_values::device_buffer_sz_sff,
        size_t _ss = default_values::seed_sz, 
        size_t _n_hash_mask_table = default_values::HASH_MASK_TABLE_SZ) : 
        Base_Class(0, 0), // the sff will manually managed its memory.
        hash_mask_table_sz(_n_hash_mask_table), times(_times)
    {

#ifdef QSKETCH_DEBUG
        std::cout << "Sketch_GPU_SF_Host_Sub_Warp_Fly: " << "_m : " << (_m) << "times: " << _times << std::endl;
        if (times <= 0) {
            std::cerr << "error : times should be greater than 0" << std::endl; // for now
        }
#endif

        hht_insert.resize(_n, _m * times);
        hht_search.resize(_n, _m);

        dht_insert.resize(device_buffer_sz / (times + 1) / (_m), (_m * times));
        dht_insert.resize(device_buffer_sz / (times + 1) / (_m), (_m));

        ds.resize(1 + default_values::HASH_MASK_ONES, _ss); // each thread only need one seed
        hash_mask_table = gpu_tool<uchar>.zero(hash_mask_table, hash_mask_table_sz * _m);
        uchar *hash_mask_table_host = generate_hashmask(nullptr, hash_mask_table_sz, _m,
            default_values::HASH_MASK_ONES, 0, &index_hash_mask_table_host);
        CUDA_CALL(cudaMemcpy(hash_mask_table, hash_mask_table_host, sizeof(uchar) * _n_hash_mask_table * _m,
            cudaMemcpyHostToDevice));
        delete []hash_mask_table_host;
    }

/*    void copy_seed_from_device()
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
*/

/*
    virtual int insert_cpu(Key_T *keys, size_t keys_sz) {
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
*/
    int copy_table_to_device(size_t b, size_t e) {

        if (e <= b) {
            return 0;
        }

        if (e - b > dht_insert.n || e - b > dht_search.n) {
            return -1;
        }

        size_t imsz = hht_insert.m * sizeof(Count_T); // insert_m_sz
        size_t smsz = hht_search.m * sizeof(Count_T);
        size_t icpd = (e - b) * imsz;
        size_t scpd = (e - b) * smsz;

        CUDA_CALL(cudaMemcpy(hht_insert.table + b * imsz, dht_insert.table, 
            icpd, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(hht_search.table + b * smsz, dht_search.table, 
            scpd, cudaMemcpyHostToDevice));
        return 0;
    }

    int copy_table_to_host(size_t b, size_t e) {
    
        if (e <= b) {
            return 0;
        }

        if (e - b < dht_insert.n || e - b < dht_search.n) {
            return -1;
        }

        size_t imsz = hht_insert.m * sizeof(Count_T); // insert_m_sz
        size_t smsz = hht_search.m * sizeof(Count_T);
        size_t icpd = (e - b) * imsz;
        size_t scpd = (e - b) * smsz;

        CUDA_CALL(cudaMemcpy(hht_insert.table + b * imsz, dht_insert.table, 
            icpd, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(hht_search.table + b * smsz, dht_search.table, 
            scpd, cudaMemcpyHostToDevice));
        return 0;

    }

    virtual int insert(Key_T *keys, size_t keys_sz) {

        //todo: keys are in host memory.

        using namespace QSKETCH_KERNEL_NAMESPACE;
        
        if (mode != Mode::INSERT) {
            return -1; // flag error
        }

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
        return 0;
    }

    virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {

        if (mod != Mode::SEARCH) {
            return -1;
        }

        using namespace QSKETCH_KERNEL_NAMESPACE;
        size_t work_load_per_warp = ceil<size_t>(keys_sz, Base_Class::nwarps);
        CUDA_CALL((search_kernel<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>
            <<<Base_Class::gridDim, Base_Class::blockDim>>>(
                keys, keys_sz,
                dht_low,
                prime_number,
                Hash_Function(),
                count, std::numeric_limits<Count_T>::max(),
                hash_mask_table,
                hash_mask_table_sz,
                work_load_per_warp,
                nullptr
            )));
        cudaDeviceSynchronize();

        return 0;
    }

    virtual void clear() {
        dht_search.clear();
        dht_insert.clear();
    }

    virtual ~Sketch_GPU_SF_Host_Sub_Warp_Fly() {
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