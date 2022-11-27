#pragma once

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
            ifs.close();
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

    T *random_freq(T *keys, size_t keys_sz, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max(),
        Freq_distribution freq = Freq_distribution()) {
        if (keys == nullptr) {
            keys = new T[keys_sz];
        }

        std::uniform_int_distribution<T> dis(min, max);
        std::uniform_int_distribution<size_t> dis_low(freq.low_freq_min, freq.low_freq_max);
        std::uniform_int_distribution<size_t> dis_high(freq.high_freq_min, freq.high_freq_max);
        std::uniform_real_distribution<float> dis_freq(0.0, 1.0);
        // std::cout << "high_freq_min: " << freq.high_freq_min << std::endl;
        // std::cout << "high_freq_max: " << freq.high_freq_max << std::endl;
        size_t i = 0;
        while (i < keys_sz) {
            T rk = dis(eng);
            size_t c;
            if (dis_freq(eng) < freq.low_freq) {
                c = dis_low(eng);
                // std::cout << "dis_low: " << c << std::endl;
            } else {
                c = dis_high(eng);
                // std::cout << "dis_high: " << c << std::endl;
            }
            size_t j = 0;
            for (; i < keys_sz && j < c; ++j, ++i) {
                keys[i] = rk;
            }
        }
        
        std::random_shuffle(keys, keys + keys_sz);

        return keys;
    }

    // T *random_freq(T *keys, size_t keys_sz, const Freq_distribution<T> &, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max(),
    //     float low_freq = 0.8, size_t low_freq_min = 1, size_t low_freq_max = 64, size_t high_freq_min =100, size_t high_freq_max = 1000) {
    //     if (keys == nullptr) {
    //         keys = new T[keys_sz];
    //     }

    //     std::uniform_int_distribution<T> dis(min, max);
    //     std::uniform_int_distribution<size_t> dis_low(low_freq_min, low_freq_max);
    //     std::uniform_int_distribution<size_t> dis_high(high_freq_min, high_freq_max);
    //     std::uniform_real_distribution<float> dis_freq(0.0, 1.0);
    //     std::cout << "high_freq_min: " << high_freq_min << std::endl;
    //     std::cout << "high_freq_max: " << high_freq_max << std::endl;
    //     size_t i = 0;
    //     while (i < keys_sz) {
    //         T rk = dis(eng);
    //         size_t c;
    //         if (dis_freq(eng) < low_freq) {
    //             c = dis_low(eng);
    //             // std::cout << "dis_low: " << c << std::endl;
    //         } else {
    //             c = dis_high(eng);
    //             // std::cout << "dis_high: " << c << std::endl;
    //         }
    //         size_t j = 0;
    //         for (; i < keys_sz && j < c; ++j, ++i) {
    //             keys[i] = rk;
    //         }
    //     }
        
    //     // std::random_shuffle(keys, keys + keys_sz);

    //     return keys;
    // }


    T *zero(T *keys, size_t keys_sz) {
        if (keys == nullptr) {
            keys = new T[keys_sz];
        }
        memset(keys, 0, sizeof(T) * keys_sz);
        return keys;
    }

    void random_shuffle(T *keys, size_t keys_sz) {
        std::random_shuffle(keys, keys + keys_sz);
    }
    void free(T *ptr) {
        delete []ptr;
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

template <typename T>
CPU_Tools<T> cpu_tool;

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
        // static CPU_Tools<Seed_T> tool;
        cpu_tool<Seed_T>.random(seed, seed_total_sz, 1, std::numeric_limits<Seed_T>::max());
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

    virtual ~Count_Min_Sketch_CPU() {
        delete [] table;
        delete [] seed;
        delete [] hs;
    }
};

extern const size_t default_hash_n;
extern const size_t default_hash_m;

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, 
    typename Hash_Function> //
struct Count_Min_Sketch_GPU_Host_Sim : Sketch<Key_T, Count_T> {

    using Insert_Kernel_Function = void (Key_T *, size_t,
        Count_T *, size_t , size_t,
        const Hash_Function &,
        Seed_T *, size_t,
        size_t,
        void *);

    using Search_Kernel_Function = void (Key_T *, size_t,
        Count_T *, size_t , size_t,
        const Hash_Function &,
        Seed_T *, size_t,
        Count_T *, Count_T,
        size_t,
        void *);

    dim3 gridDim;
    dim3 blockDim;

    size_t nwarps;
    size_t nthreads;

    // Hash_Function *hash = nullptr;
    Insert_Kernel_Function *insert_kernel = nullptr;
    Search_Kernel_Function *search_kernel = nullptr;

    size_t n;
    size_t m;
    size_t table_total_sz;
    Count_T *table = nullptr;

    size_t seed_sz;
    size_t seed_num;
    size_t seed_total_sz;

    Seed_T *seed = nullptr;

    Count_Min_Sketch_GPU_Host_Sim(
        Insert_Kernel_Function *_insert_kernel,
        Search_Kernel_Function *_search_kernel,
        size_t _n = default_hash_n, size_t _m = default_hash_m, size_t ss = default_seed_sz) : 
        n(_n), m(_m),
        seed_sz(ss) {

        insert_kernel = _insert_kernel;
        search_kernel = _search_kernel;

        seed_num = m * WARP_SIZE + 1;
        // std::cout << "seed_num: " << seed_num << std::endl;

        table_total_sz = n * m * WARP_SIZE;
        // std::cout << "table_total_sz: " << table_total_sz << std::endl;
        table = cpu_tool<Count_T>.zero(table, table_total_sz);

        seed_total_sz = seed_sz * seed_num;
        // std::cout << "seed_total_sz: " << seed_total_sz << std::endl;

        // random seeds
        seed = cpu_tool<Seed_T>.random(seed, seed_total_sz, 1, std::numeric_limits<Seed_T>::max());
        gridDim = default_grid_dim;
        blockDim = default_block_dim;

        nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
        nwarps = ceil<size_t>(nthreads, WARP_SIZE);

    }

    virtual int insert(Key_T *keys, size_t keys_sz) {
        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        insert_kernel(
                keys, keys_sz,
                table, n, m,
                Hash_Function(),
                seed, seed_sz,
                work_load_per_warp,
                nullptr
            );
        return 0;
    }
    virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        search_kernel(
                keys, keys_sz,
                table, n, m,
                Hash_Function(),
                seed, seed_sz,
                count, std::numeric_limits<Count_T>::max(),
                work_load_per_warp,
                nullptr
            );
        return 0;
    }

    virtual void clear() {
        table = cpu_tool<Count_T>.zero(table, table_total_sz);
    }

    virtual ~Count_Min_Sketch_GPU_Host_Sim() {
        delete [] table;
        delete [] seed;
    }
};
