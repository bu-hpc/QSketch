#pragma once

namespace qsketch {


template <typename Key_T, typename Count_T> // 
struct Unordered_Map_GPU : Unordered_Map<Key_T, Count_T> {
    size_t host_buf_sz;
    Key_T *keys_buf = nullptr;
    Count_T *counts_buf = nullptr;
    Unordered_Map_GPU(size_t _host_buf_sz = 1 << 20) : host_buf_sz(_host_buf_sz) {
        keys_buf = new Key_T[host_buf_sz];
        counts_buf = new Count_T[host_buf_sz];
    }

    virtual int insert(Key_T *keys, size_t keys_sz) {
        for (size_t i = 0; i < keys_sz; i += host_buf_sz) {
            size_t cpd = std::min(host_buf_sz, keys_sz - i);
            CUDA_CALL(cudaMemcpy(keys_buf, keys + i, sizeof(Key_T) * cpd, cudaMemcpyDeviceToHost));
            Unordered_Map<Key_T, Count_T>::insert(keys_buf, cpd);
        }
        return 0;
    }
    virtual int search(Key_T *keys, size_t keys_sz, Count_T *counts) {
        for (size_t i = 0; i < keys_sz; i += host_buf_sz) {
            size_t cpd = std::min(host_buf_sz, keys_sz - i);
            CUDA_CALL(cudaMemcpy(keys_buf, keys + i, sizeof(Key_T) * cpd, cudaMemcpyDeviceToHost));
            Unordered_Map<Key_T, Count_T>::search(keys_buf + i, cpd, counts_buf);
#if 0
            for (size_t j = 0; j < cpd; ++j) {
                std::cout << count_buf[j] << std::endl;
            }
#endif
            CUDA_CALL(cudaMemcpy(counts + i, counts_buf, sizeof(Key_T) * cpd, cudaMemcpyHostToDevice));
        }
        return 0;
    }



    virtual size_t get_counts(Key_T *keys, size_t keys_sz, Count_T *counts) {
        size_t i = 0;

        while (i < keys_sz) {
            size_t cpd = std::min(host_buf_sz, keys_sz - i);
            size_t rd = Unordered_Map<Key_T, Count_T>::get_counts(keys_buf, cpd, counts_buf);
            if (rd == 0) {
                break;
            }
            CUDA_CALL(cudaMemcpy(keys + i, keys_buf, sizeof(Key_T) * rd, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(counts + i, counts_buf, sizeof(Count_T) * rd, cudaMemcpyHostToDevice));
            i += rd;
        }
        return i;
    }

    virtual ~Unordered_Map_GPU() {
        delete []keys_buf;
        delete []counts_buf;
    }

#ifdef QSKETCH_DEBUG
    
#endif
};

template <typename Count_T>
struct Device_Hash_Table {
    size_t n = 0; // the number of hash tables
    size_t m = 0; // the size of each hash table
    size_t table_total_sz = 0;
    Count_T *table = nullptr;
    uint *next_level_id = nullptr;

    Device_Hash_Table() = default;
    Device_Hash_Table(size_t _n, size_t _m) {
#ifdef QSKETCH_DEBUG
        // std::cout << "Device_Hash_Table" << std::endl;
#endif
        resize(_n, _m);
    }

    void resize(size_t _n, size_t _m) {
        if (_n != n || _m != m) {
            if (table != nullptr) {
                cudaFree(table);
                table = nullptr;
#ifdef QSKETCH_DEBUG
                total_memory_usage -= sizeof(Count_T) * table_total_sz;
#endif
            }
            n = _n;
            m = _m;
            table_total_sz = n * m;

            // std::cout << "p1" << std::endl;
            table = gpu_tool<Count_T>.zero(table, table_total_sz);
            // std::cout << "p2" << std::endl;


#ifdef QSKETCH_DEBUG
                total_memory_usage += sizeof(Count_T) * table_total_sz;
                #if 0
                std::cout << format_memory_usage<unsigned char>(total_memory_usage) << std::endl;
                #endif
#endif
            next_level_id = gpu_tool<uint>.zero(next_level_id, 1);
            {
                uint next_level_id_ini_val = default_values::NEXT_LEVEL_ID_START + 1;
                CUDA_CALL(cudaMemcpy(next_level_id, &next_level_id_ini_val, sizeof(uint), cudaMemcpyHostToDevice));
            }
        }
        
    }
    void clear() {
        if (table_total_sz == 0) {
            return;
        }
        table = gpu_tool<Count_T>.zero(table, table_total_sz);
        {
            uint next_level_id_ini_val = default_values::NEXT_LEVEL_ID_START + 1;
            CUDA_CALL(cudaMemcpy(next_level_id, &next_level_id_ini_val, sizeof(uint), cudaMemcpyHostToDevice));
        }
    }

    __host__ void free() {
        n = 0;
        m = 0;
        table_total_sz = 0;
        if (table != nullptr) {
            cudaFree(table);
            table = nullptr;
        }
        if (next_level_id != nullptr) {
            cudaFree(next_level_id);
            next_level_id = nullptr;
        }
    }
// #ifdef QSKETCH_DEBUG
    void print_next_level_id() {
        uint h_nli;
        CUDA_CALL(cudaMemcpy(&h_nli, next_level_id, sizeof(uint), cudaMemcpyDeviceToHost));
        std::cout << "next_level_id: " << h_nli << std::endl;
    }
// #endif

    operator bool() {
        return table_total_sz;
    }
};

template <typename Count_T>
struct Managed_Hash_Table {
    size_t n = 0; // the number of hash tables
    size_t m = 0; // the size of each hash table
    size_t table_total_sz = 0;
    Count_T *table = nullptr;

    Managed_Hash_Table() = default;
    Managed_Hash_Table(size_t _n, size_t _m) {
#ifdef QSKETCH_DEBUG
        // std::cout << "Device_Hash_Table" << std::endl;
#endif
        resize(_n, _m);
    }

    void resize(size_t _n, size_t _m) {
        if (_n != n || _m != m) {
            if (table != nullptr) {
                cudaFree(table);
                table = nullptr;
#ifdef QSKETCH_DEBUG
                total_memory_usage -= sizeof(Count_T) * table_total_sz;
#endif
            }
            n = _n;
            m = _m;
            table_total_sz = n * m;

            // std::cout << "p1" << std::endl;
            table = gpu_tool<Count_T>.zero_managed(table, table_total_sz);
            // std::cout << "p2" << std::endl;

        }
        
    }
    void clear() {
        if (table_total_sz == 0) {
            return;
        }
    }

    __host__ void free() {
        n = 0;
        m = 0;
        table_total_sz = 0;
        if (table != nullptr) {
            cudaFree(table);
            table = nullptr;
        }
    }

    operator bool() {
        return table_total_sz;
    }
};


template <typename Count_T>
struct Zero_Copy_Hash_Table {
    size_t n = 0; // the number of hash tables
    size_t m = 0; // the size of each hash table
    size_t table_total_sz = 0;
    Count_T *table = nullptr;
    Count_T *host_table = nullptr;
    // uint *next_level_id = nullptr;

    Zero_Copy_Hash_Table() = default;
    Zero_Copy_Hash_Table(size_t _n, size_t _m) {
#ifdef QSKETCH_DEBUG
        // std::cout << "Device_Hash_Table" << std::endl;
#endif
        resize(_n, _m);
    }

    int resize(size_t _n, size_t _m) {
        if (_n != n || _m != m) {
            if (table != nullptr) {
                // cudaFree(table);
                cudaFreeHost(host_table);
                table = nullptr;
                host_table = nullptr;
            }
            n = _n;
            m = _m;
            table_total_sz = n * m;
            // table = gpu_tool<Count_T>.zero(table, table_total_sz);
            gpu_tool<Count_T>.zero_zero_copy_memory(&host_table, &table, table_total_sz);
            // cudaDeviceSynchronize();
            // qsketch::check_cuda_error();
            // CUDA_CALL(cudaDeviceSynchronize());
        }
        return 0;
    }
    void clear() {
        if (table_total_sz == 0) {
            return;
        }
        table = gpu_tool<Count_T>.zero(table, table_total_sz);
    }

    __host__ void free() {
        n = 0;
        m = 0;
        table_total_sz = 0;
        if (table != nullptr) {
            cudaFreeHost(host_table);
            // std::cout << "free" << std::endl;
            table = nullptr;
            host_table = nullptr;
        }
    }
    operator bool() {
        return table_total_sz;
    }
};

template <typename Seed_T>
struct Device_Seed {
    size_t seed_num = 0;
    size_t seed_sz = 0;
    size_t seed_total_sz = 0;
    Seed_T *seed = nullptr;

    Device_Seed() = default;
    // Device_Seed(const Device_Seed<Seed_T> &) = default;
    Device_Seed(size_t _seed_num, size_t _seed_sz = 1) {
        resize(_seed_num, _seed_sz);
    }
    void resize(size_t _seed_num, size_t _seed_sz = 1) {
        if (_seed_num != seed_num || _seed_sz != seed_sz) {
            seed_num = _seed_num;
            seed_sz = _seed_sz;
            seed_total_sz = seed_num * seed_sz;
            if (seed != nullptr) {
                cudaFree(seed);
                seed = nullptr;
            }
            if (seed_total_sz != 0)
                seed = gpu_tool<Seed_T>.random(seed, seed_total_sz, 1, std::numeric_limits<Seed_T>::max());

            #ifdef QSKETCH_DEBUG
                    std::cout << "seed: " << std::endl;
                    Seed_T *s = new Seed_T[seed_total_sz];
                    CUDA_CALL(cudaMemcpy(s, seed, sizeof(Seed_T) * seed_total_sz, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < seed_total_sz; ++i) {
                        std::cout << s[i] << ", ";
                    }
                    std::cout << std::endl;
            #endif
        }
    }

    __host__ void free() {

        /*
            This function must be called before the destructor.
            And it must be called from the host.
        */

        seed_num = 0;
        seed_sz = 0;
        seed_total_sz = 0;
        if (seed != nullptr) {
            cudaFree(seed);
            seed = nullptr;
        }
    }
};

template <typename T = uchar>
struct Hashmask_Table
{
    const T *device_table;
    __device__ __host__ bool get(uint id) const{
        return false;

    }
    __device__ __host__ bool set(uint id) {
        return false;
    }
};

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, 
    typename Hash_Function> // 
struct Sketch_GPU : public Sketch<Key_T, Count_T> {

    dim3 gridDim;
    dim3 blockDim;

    size_t nwarps;
    size_t nthreads;

    size_t prime_number; // a large prime number

    Device_Hash_Table<Count_T> dht_low;
    Device_Hash_Table<Count_T> dht_high;
    Device_Seed<Seed_T> ds;

    uchar *hash_mask_table = nullptr; // device hash_mask_table

    Sketch_GPU(size_t _n_low, size_t _m_low, size_t _n_high = 0, size_t _m_high = 0)
        : dht_low(_n_low, _m_low), dht_high(_n_high, _m_high) {

#ifdef QSKETCH_DEBUG
        std::cout << "Sketch_GPU" << std::endl;
#endif

        gridDim = default_values::grid_dim;
        blockDim = default_values::block_dim;
        nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
        nwarps = ceil<size_t>(nthreads, default_values::WARP_SIZE);
        prime_number = default_values::PRIME_NUMBER;

#ifdef QSKETCH_DEBUG
        show_memory_usage(std::cout);
#endif

    }

    static size_t number_of_buckets(size_t insert_keys_sz, size_t m, double factor) {
        size_t table_sz_row = ((insert_keys_sz / factor) * default_values::HASH_MASK_ONES) / m;
        size_t table_sz_prime = find_greatest_prime(table_sz_row);
        // std::cout << "table_sz: " << table_sz_row << ", " << table_sz_prime  << std::endl;
        return table_sz_prime;
    }

    virtual void clear() {
        dht_low.clear();
        dht_high.clear();
    }

    // virtual size_t number_of_buckets(size_t insert_keys_sz, size_t m, double factor) {
    //     return 0;
    // }

    virtual ~Sketch_GPU() {
        // cudaFree(table);
        // cudaFree(seed);
        dht_low.free();
        dht_high.free();
        ds.free();
        cudaFree(hash_mask_table);
    }

#ifdef QSKETCH_DEBUG
    virtual void print(std::ostream &os) {

    }


#endif
    virtual void show_memory_usage(std::ostream &os) {
        size_t table_total_sz = dht_low.table_total_sz + dht_high.table_total_sz;
        // os << "memory_usage: " << format_memory_usage<Count_T>(table_total_sz) << std::endl;
        os << format_memory_usage<Count_T>(table_total_sz) << std::endl;
    }

};

}

#include "device_hash_function.h"
#include "device_sketch_v0.h"
#include "device_sketch_v01.h"
#include "device_sketch_v1.h"
#include "device_sketch_v2.h"
#include "device_sketch_v3.h"
// #include "device_sketch_v31.h"
#include "device_sketch_v32.h"
#include "device_sketch_v4.h"
#include "device_sketch_v41.h"
#include "device_sketch_v42.h"
//#include "device_sketch_v5.h"
