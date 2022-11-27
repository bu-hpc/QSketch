#pragma once

#include "host_hash_function.h"

namespace qsketch {

extern const size_t default_hash_table_sz;
extern const size_t default_hash_table_num;
extern const size_t default_seed_sz;
extern const size_t defualt_seed_num;

extern const size_t default_hash_n;
extern const size_t default_hash_m;

template <typename Key_T, typename Count_T> // 
struct Unordered_Map : Sketch<Key_T, Count_T> {
    std::unordered_map<Key_T, Count_T> um;
    typename std::unordered_map<Key_T, Count_T>::iterator it;
    size_t limit = std::numeric_limits<size_t>::max();
    virtual int insert(Key_T *keys, size_t keys_sz) {
        for (size_t i = 0; i < keys_sz; ++i) {
            if (um.size() < limit) {
                ++um[keys[i]];                
            } else {
                auto fd = um.find(keys[i]);
                if (fd != um.end()) {
                    fd->second++;
                }
            }
        }
        it = um.begin();
        return 0;
    }
    virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
        for (size_t i = 0; i < keys_sz; ++i) {
            count[i] = um[keys[i]];
        }
        return 0;
    }
    virtual void clear() {
        um.clear();
    }

    virtual size_t get_counts(Key_T *keys, size_t keys_sz, Count_T *counts) {
        size_t i = 0;
        while (it != um.end() && i < keys_sz) {
            keys[i] = it->first;
            counts[i] = it->second;
            ++it;
            ++i;
        }
        return i;
    }
#ifdef DEBUG
    virtual void print(std::ostream &os) {

    }

    virtual void show_memory_usage(std::ostream &os) {
        os << "memory_usage: " << format_memory_usage<pair<Key_T, Count_T>>(um.size()) << std::endl;
    }
#endif
};



template <typename Count_T>
Count_T add_min(std::vector<Count_T *> v, const Count_T &val) {
    std::sort(v.begin(), v.end(), [](Count_T *p1, Count_T *p2) {
        return (*p1 < * p2);
    });
    Count_T min = *v[0];
    for (size_t i = 0; i < v.size() && val != 0; ++i) {
        if (*v[i] == min) {
            ++(*v[i]);
        } else {
            break;
        }
    }
    return min;
}

// template <typename Key_T, typename Count_T, typename Hashed_T = size_t, typename Seed_T = size_t> // 
// struct Count_Min_Sketch_CPU : Sketch<Key_T, Count_T> {
//     size_t hash_table_sz;
//     size_t hash_table_num;
//     size_t table_total_sz;
//     Count_T *table = nullptr;

//     size_t seed_sz;
//     size_t seed_num;
//     size_t seed_total_sz;
//     Seed_T *seed = nullptr;
//     Hash<Seed_T> *hs = nullptr;

//     Count_Min_Sketch_CPU(size_t hts = default_hash_table_sz, size_t htn = default_hash_table_num,
//         size_t ss = default_seed_sz, size_t sn = defualt_seed_num) : hash_table_sz(hts), hash_table_num(htn), seed_sz(ss), seed_num(sn) {

//         table_total_sz = hash_table_sz * hash_table_num;
//         table = new Count_T[table_total_sz];
//         memset(table, 0, sizeof(Count_T) * table_total_sz);

//         seed_total_sz = seed_sz * seed_num;
//         seed = new Seed_T[seed_total_sz];

//         cpu_tool<Seed_T>.random(seed, seed_total_sz, 1, std::numeric_limits<Seed_T>::max());
//         hs = new Hash<Seed_T>[seed_num];
//         for (size_t i = 0; i < seed_num; ++i) {
//             hs[i].set_seed(seed + seed_sz * i, seed_sz);
//         }

//     }

//     Count_T insert(const Key_T &key) {

//         std::vector<Count_T *> v(hash_table_num);
//         for (size_t i = 0; i < hash_table_num; ++i) {
//             Hashed_T thv = hs[i].hash_mul_add(key) % hash_table_sz;
//             v[i] = table + hash_table_sz * i + thv;
//         }
//         return add_min(v, Count_T(1));
//     }

//     Count_T search(const Key_T &key) const {
//         std::vector<Count_T *> v(hash_table_num);
//         for (size_t i = 0; i < hash_table_num; ++i) {
//             Hashed_T thv = hs[i].hash_mul_add(key) % hash_table_sz;
//             v[i] = table + hash_table_sz * i + thv;
//         }
//         return add_min(v, Count_T(1));
//     }

//     virtual int insert(Key_T *keys, size_t keys_sz) {
//         for (size_t i = 0; i < keys_sz; ++i) {
//             insert(keys[i]);
//         }

//         return 0;
//     }
//     virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {

//         if (count == nullptr) {
//             count = new Count_T[keys_sz];
//         }

//         for (size_t i = 0; i < keys_sz; ++i) {
//             count[i] = search(keys[i]);
//         }

//         return 0;
//     }

//     virtual void clear() {
//         memset(table, 0, sizeof(Count_T) * table_total_sz);
//     }

//     virtual void print(std::ostream &os) {
//         for (size_t i = 0; i < table_total_sz; ++i) {
//             os << table[i] << std::endl;
//         }
//     }

//     virtual ~Count_Min_Sketch_CPU() {
//         delete [] table;
//         delete [] seed;
//         delete [] hs;
//     }
// };

template <typename Count_T>
struct Atomic_Hash_Table {
    size_t n = 0; // the number of hash tables
    size_t m = 0; // the size of each hash table
    size_t table_total_sz = 0;
    std::atomic<Count_T> *table = nullptr;

    Atomic_Hash_Table() = default;
    Atomic_Hash_Table(size_t _n, size_t _m) {
#ifdef QSKETCH_DEBUG
        // std::cout << "Host_Hash_Table" << std::endl;
#endif
        resize(_n, _m);
    }

    void resize(size_t _n, size_t _m) {
        if (_n != n || _m != m) {
            if (table != nullptr) {
                // cudaFree(table);
                delete [] table;
                // cpu_tool<Count_T>.free(table);
                table = nullptr;
            }
            n = _n;
            m = _m;
            table_total_sz = n * m;
            // table = cpu_tool<Count_T>.zero(table, table_total_sz);
            table = new std::atomic<Count_T>[table_total_sz];
            for (size_t i = 0; i < table_total_sz; ++i) {
                table[i] = Count_T(0);
            }
        }
        
    }
    void clear() {
        if (table_total_sz == 0) {
            return;
        }
        // table = cpu_tool<Count_T>.zero(table, table_total_sz);
        table = new std::atomic<Count_T>[table_total_sz];
        for (size_t i = 0; i < table_total_sz; ++i) {
            // std::cout << i << std::endl;
            table[i] = Count_T(0);
        }
    }

    void free() {
        n = 0;
        m = 0;
        table_total_sz = 0;
        if (table != nullptr) {
            // cudaFree(table);
            // cpu_tool<Count_T>.free(table);
            delete [] table;
            table = nullptr;
        }
    }

    operator bool() {
        return table_total_sz;
    }
};


template <typename Count_T>
struct Host_Hash_Table {
    size_t n = 0; // the number of hash tables
    size_t m = 0; // the size of each hash table
    size_t table_total_sz = 0;
    Count_T *table = nullptr;
    uint *next_level_id = nullptr;

    Host_Hash_Table() = default;
    Host_Hash_Table(size_t _n, size_t _m) {
#ifdef QSKETCH_DEBUG
        // std::cout << "Host_Hash_Table" << std::endl;
#endif
        resize(_n, _m);
    }

    void resize(size_t _n, size_t _m) {
        if (_n != n || _m != m) {
            if (table != nullptr) {
                // cudaFree(table);
                // delete table;
                cpu_tool<Count_T>.free(table);
                table = nullptr;
            }
            n = _n;
            m = _m;
            table_total_sz = n * m;
            table = cpu_tool<Count_T>.zero(table, table_total_sz);
        }
        
    }
    void clear() {
        if (table_total_sz == 0) {
            return;
        }
        table = cpu_tool<Count_T>.zero(table, table_total_sz);
    }

    void free() {
        n = 0;
        m = 0;
        table_total_sz = 0;
        if (table != nullptr) {
            // cudaFree(table);
            cpu_tool<Count_T>.free(table);
            table = nullptr;
        }
        if (next_level_id != nullptr) {
            cpu_tool<Count_T>.free(next_level_id);
            next_level_id = nullptr;
        }
    }
#ifdef QSKETCH_DEBUG
    void print_next_level_id() {
        uint h_nli;
        CUDA_CALL(cudaMemcpy(&h_nli, next_level_id, sizeof(uint), cudaMemcpyDeviceToHost));
        std::cout << "next_level_id: " << h_nli << std::endl;
    }
#endif

    operator bool() {
        return table_total_sz;
    }
};

template <typename Seed_T>
struct Host_Seed {
    size_t seed_num = 0;
    size_t seed_sz = 0;
    size_t seed_total_sz = 0;
    Seed_T *seed = nullptr;

    Host_Seed() = default;
    Host_Seed(const Host_Seed &) = default;
    Host_Seed(Host_Seed &&) = default;
    
    // Device_Seed(const Device_Seed<Seed_T> &) = default;
    Host_Seed(size_t _seed_num, size_t _seed_sz = 1) {
        resize(_seed_num, _seed_sz);
    }

    // template <typename DS>
    // Host_Seed(const DS &ds) 
    // {
    //     seed_num = ds.seed_num;
    //     seed_sz = ds.seed_sz;
    //     seed_total_sz = ds.seed_total_sz;
    //     if (seed) {
    //         delete [] seed;
    //     }
    //     seed = cpu_tool<Seed_T>.zero(nullptr, seed_total_sz);
    //     CUDA_CALL(cudaMemcpy(seed, ds.seed, sizeof(Seed_T) * seed_total_sz, cudaMemcpyDeviceToHost));
    //     std::cout << "host seed: ";
    //     for (size_t i = 0; i < seed_total_sz; ++i) {
    //         std::cout << seed[i] << ",";
    //     }
    //     std::cout << std::endl;
    // }
    // template <typename DS>
    // void copy_from_device(const DS &ds)
    // {
    //     seed_num = ds.seed_num;
    //     seed_sz = ds.seed_sz;
    //     seed_total_sz = ds.seed_total_sz;
    //     if (seed) {
    //         delete [] seed;
    //     }
    //     seed = cpu_tool<Seed_T>.zero(nullptr, seed_total_sz);
    //     CUDA_CALL(cudaMemcpy(seed, ds.seed, sizeof(Seed_T) * seed_total_sz, cudaMemcpyDeviceToHost));
    //     return *this;
    // }

    void resize(size_t _seed_num, size_t _seed_sz = 1) {
        if (_seed_num != seed_num || _seed_sz != seed_sz) {
            seed_num = _seed_num;
            seed_sz = _seed_sz;
            seed_total_sz = seed_num * seed_sz;
            if (seed != nullptr) {
                // cudaFree(seed);
                cpu_tool<Seed_T>.free(seed);
                seed = nullptr;
            }
            if (seed_total_sz != 0)
                seed = cpu_tool<Seed_T>.random(seed, seed_total_sz, 1, std::numeric_limits<Seed_T>::max());

            #ifdef QSKETCH_DEBUG
                    // std::cout << "seed: " << std::endl;
                    // Seed_T *s = new Seed_T[seed_total_sz];
                    // CUDA_CALL(cudaMemcpy(s, seed, sizeof(Seed_T) * seed_total_sz, cudaMemcpyDeviceToHost));
                    // for (size_t i = 0; i < seed_total_sz; ++i) {
                    //     std::cout << s[i] << ", ";
                    // }
                    // std::cout << std::endl;
            #endif
        }
    }

    void free() {

        /*
            This function must be called before the destructor.
            And it must be called from the host.
        */

        seed_num = 0;
        seed_sz = 0;
        seed_total_sz = 0;
        if (seed != nullptr) {
            cpu_tool<Seed_T>.free(seed);
            seed = nullptr;
        }
    }
};

}

#include "host_sketch_simulator.h"
