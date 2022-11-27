#pragma once
#include "cpu_hash_functions.h"

constexpr size_t seed_buf_sz = 128;

template <typename T>
struct CPU_Tools : Tools<T> {
    std::default_random_engine eng;
    char seed_buf[seed_buf_sz];
    CPU_Tools(bool random_seed = true) {
        if (random_seed) {
            std::ifstream ifs("/dev/urandom");
            ifs.read(seed_buf, seed_buf_sz);
            std::seed_seq sq(seed_buf, seed_buf + seed_buf_sz);
            eng.seed(sq);
        }
    }

    T *random(T *keys, size_t keys_sz, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max()) {

        if (keys == nullptr) {
            keys = new T[keys_sz];
        }

        std::uniform_int_distribution<T> dis(min, max);

        for (size_t i = 0; i < keys_sz; ++i)
        {
            keys[i] = dis(eng);
        }

        return keys;
    }
    T *zero(T *keys, size_t keys_sz) {
        if (keys == nullptr) {
            keys = new T[keys_sz];
        }
        memset(keys, 0, sizeof(T) * keys_sz);
    }
};

template <typename Key_T, typename Count_T, typename Hashed_T = size_t> // 
struct Unordered_Map : Sketch<Key_T, Count_T> {
    std::unordered_map<Key_T, Count_T> um;
    virtual int insert(Key_T *keys, size_t keys_sz) {
        for (size_t i = 0; i < keys_sz; ++i) {
            ++um[keys[i]];
        }
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
};


extern const size_t default_hash_table_sz;
extern const size_t default_hash_table_num;
extern const size_t default_seed_sz;
extern const size_t defualt_seed_num;

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

template <typename Key_T, typename Count_T, typename Hashed_T = size_t, typename Seed_T = size_t> // 
struct Count_Min_Sketch_CPU : Sketch<Key_T, Count_T> {
    size_t hash_table_sz;
    size_t hash_table_num;
    size_t table_total_sz;
    Count_T *table = nullptr;

    size_t seed_sz;
    size_t seed_num;
    size_t seed_total_sz;
    Seed_T *seed = nullptr;
    Hash<Seed_T> *hs = nullptr;

    Count_Min_Sketch_CPU(size_t hts = default_hash_table_sz, size_t htn = default_hash_table_num,
        size_t ss = default_seed_sz, size_t sn = defualt_seed_num) : hash_table_sz(hts), hash_table_num(htn), seed_sz(ss), seed_num(sn) {

        table_total_sz = hash_table_sz * hash_table_num;
        table = new Count_T[table_total_sz];
        memset(table, 0, sizeof(Count_T) * table_total_sz);

        seed_total_sz = seed_sz * seed_num;
        seed = new Seed_T[seed_total_sz];

        // random seeds
        static CPU_Tools<Seed_T> tool;
        tool.random(seed, seed_total_sz, 1, std::numeric_limits<Seed_T>::max());
        hs = new Hash<Seed_T>[seed_num];
        for (size_t i = 0; i < seed_num; ++i) {
            hs[i].set_seed(seed + seed_sz * i, seed_sz);
        }

    }

    Count_T insert(const Key_T &key) {

        std::vector<Count_T *> v(hash_table_num);
        for (size_t i = 0; i < hash_table_num; ++i) {
            Hashed_T thv = hs[i].hash_mul_add(key) % hash_table_sz;
            v[i] = table + hash_table_sz * i + thv;
        }
        return add_min(v, Count_T(1));
    }

    Count_T search(const Key_T &key) const {
        std::vector<Count_T *> v(hash_table_num);
        for (size_t i = 0; i < hash_table_num; ++i) {
            Hashed_T thv = hs[i].hash_mul_add(key) % hash_table_sz;
            v[i] = table + hash_table_sz * i + thv;
        }
        return add_min(v, Count_T(1));
    }

    virtual int insert(Key_T *keys, size_t keys_sz) {
        // std::cout << "Count_Min_Sketch_CPU : insert" << std::endl;
        for (size_t i = 0; i < keys_sz; ++i) {
            // std::cout << "insert: " << i << std::endl;
            insert(keys[i]);
        }

        return 0;
    }
    virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {

        if (count == nullptr) {
            count = new Count_T[keys_sz];
            // memset(count, 0, sizeof(Count_T) * keys_sz);
        }

        for (size_t i = 0; i < keys_sz; ++i) {
            count[i] = search(keys[i]);
        }

        return 0;
    }

    virtual void clear() {
        memset(table, 0, sizeof(Count_T) * table_total_sz);
    }

    virtual void print(std::ostream &os) {
        for (size_t i = 0; i < table_total_sz; ++i) {
            os << table[i] << std::endl;
        }
    }
};
