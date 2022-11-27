#pragma once

const char* curandGetStatusString(curandStatus_t status);

template <typename T>
struct GPU_Tools : CPU_Tools<T> {

    curandGenerator_t gen;
    GPU_Tools(bool random_seed = true) : CPU_Tools<T>(random_seed) {}

    T *random(T *keys, size_t keys_sz, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max()) {

        // std::cout << "cpu random" << std::endl;

        T *tem = CPU_Tools<T>::random(nullptr, keys_sz, min, max);

        if (keys == nullptr) {
            checkKernelErrors(cudaMalloc(&keys, sizeof(T) * keys_sz));
        }
        checkKernelErrors(cudaMemcpy(keys, tem, sizeof(T) * keys_sz, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        // if (keys_sz == 2056) {
        //     for (size_t i = 0; i < keys_sz; ++i) {
        //         std::cout << "seed: " << tem[i] << std::endl;
        //     }
        // }
        delete []tem;
        return keys;
    }

    T *random_freq(T *keys, size_t keys_sz, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max(),
        Freq_distribution freq = Freq_distribution()) {
        T *tem = CPU_Tools<T>::random_freq(nullptr, keys_sz, min, max,
            freq);

        // std::cout << "random_freq" << std::endl;
        if (keys == nullptr) {
            checkKernelErrors(cudaMalloc(&keys, sizeof(T) * keys_sz));
        }
        checkKernelErrors(cudaMemcpy(keys, tem, sizeof(T) * keys_sz, cudaMemcpyHostToDevice));

        // if (keys_sz == 2056) 
        // {
        //     for (size_t i = 0; i < keys_sz; ++i) {
        //         std::cout << "rand: " << tem[i] << std::endl;
        //     }
        // }

        // {
        //     std::unordered_map<unsigned int, unsigned int> um;
        //     unsigned int m = 0;
        //     for (size_t i = 0; i < keys_sz; ++i) {
        //         // std::cout << tem[i] << std::endl;
        //         m = std::max(m, ++um[tem[i]]);
        //     }
        //     std::cout << "um max: " << m << std::endl;
        // }

        delete []tem;
        return keys;
    }



    // T *random_freq(T *keys, size_t keys_sz, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max(),
    //     float low_freq = 1.0, size_t low_freq_min = 1, size_t low_freq_max = 1, size_t high_freq_min =4, size_t high_freq_max = 8) {
    //     T *tem = CPU_Tools<T>::random_freq(nullptr, keys_sz, min, max,
    //         low_freq, low_freq_min, low_freq_max, high_freq_min, high_freq_max);

    //     std::cout << "random_freq" << std::endl;
    //     if (keys == nullptr) {
    //         checkKernelErrors(cudaMalloc(&keys, sizeof(T) * keys_sz));
    //     }
    //     checkKernelErrors(cudaMemcpy(keys, tem, sizeof(T) * keys_sz, cudaMemcpyHostToDevice));

    //     // if (keys_sz == 2056) {
    //     //     for (size_t i = 0; i < keys_sz; ++i) {
    //     //         std::cout << "seed: " << tem[i] << std::endl;
    //     //     }
    //     // }

    //     {
    //         std::unordered_map<unsigned int, unsigned int> um;
    //         unsigned int m = 0;
    //         for (size_t i = 0; i < keys_sz; ++i) {
    //             // std::cout << tem[i] << std::endl;
    //             m = std::max(m, ++um[tem[i]]);
    //         }
    //         std::cout << m << std::endl;
    //     }

    //     delete []tem;
    //     return keys;
    // }

    T *zero(T *keys, size_t keys_sz) {
        // std::cout << "zero 0" << std::endl;
        // std::cout << "cpu zero 0" << std::endl;

        if (keys == nullptr) {
            checkKernelErrors(cudaMalloc(&keys, sizeof(T) * keys_sz));
        }
        // std::cout << "cpu zero 1" << std::endl;
        cudaMemset(keys, 0, sizeof(T) * keys_sz);
        return keys;
    }

    void random_shuffle(T *keys, size_t keys_sz) {
        // std::random_shuffle(keys, keys + keys_sz);
    }

    void free(T *ptr) {
        cudaFree(ptr);
    }
};



/*
gpu rand
*/

template <>
struct GPU_Tools<unsigned int> : CPU_Tools<unsigned int> {
    using T = unsigned int;
    curandGenerator_t gen;
    GPU_Tools(bool random_seed = true) : CPU_Tools<T>(random_seed) {
        CURAND_CALL(curandCreateGenerator(&gen, 
            CURAND_RNG_PSEUDO_DEFAULT));
        if (DEBUG_RANDOM_SEED)
            CURAND_CALL(curandSetGeneratorOffset(gen, this->rand()));
        else 
            CURAND_CALL(curandSetGeneratorOffset(gen, 1234u));
    }

    T *random(T *keys, size_t keys_sz, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max()) {

       // std::cout << "gpu random, unsigned int, " << keys_sz << std::endl;
        if (keys == nullptr) {
            checkKernelErrors(cudaMalloc(&keys, sizeof(unsigned int) * keys_sz));
        }
        CURAND_CALL(curandGenerate(gen, keys, keys_sz));
        cudaDeviceSynchronize();

        // unsigned int *buf = new unsigned int[keys_sz];
        // checkKernelErrors(cudaMemcpy(buf, keys, sizeof(unsigned int) * keys_sz, cudaMemcpyDeviceToHost));

        // size_t count = 0;

        // for (size_t i = 0; i < keys_sz; ++i) {
        //     if (buf[i] == 0) {
        //         count++;
        //     }

        //     // if (keys_sz == 2056) 
        //     {
        //         std::cout << "seed: " << buf[i] << std::endl;
        //     }
        // }

        // std::cout << "zero seeds: " << count << std::endl;

        return keys;
    }

    T *random_freq(T *keys, size_t keys_sz, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max(),
        Freq_distribution freq = Freq_distribution()) {
        T *tem = CPU_Tools<T>::random_freq(nullptr, keys_sz, min, max,
            freq);

        // std::cout << "random_freq" << std::endl;
        if (keys == nullptr) {
            checkKernelErrors(cudaMalloc(&keys, sizeof(T) * keys_sz));
        }
        checkKernelErrors(cudaMemcpy(keys, tem, sizeof(T) * keys_sz, cudaMemcpyHostToDevice));

        // if (keys_sz == 2056) 
        // {
        //     for (size_t i = 0; i < keys_sz; ++i) {
        //         std::cout << "rand: " << tem[i] << std::endl;
        //     }
        // }

        // {
        //     std::unordered_map<unsigned int, unsigned int> um;
        //     unsigned int m = 0;
        //     for (size_t i = 0; i < keys_sz; ++i) {
        //         // std::cout << tem[i] << std::endl;
        //         m = std::max(m, ++um[tem[i]]);
        //     }
        //     std::cout << "um max: " << m << std::endl;
        // }

        delete []tem;
        return keys;
    }

    T *zero(T *keys, size_t keys_sz) {

        // std::cout << "zero 1" << std::endl;

        if (keys == nullptr) {
            checkKernelErrors(cudaMalloc(&keys, sizeof(T) * keys_sz));
        }
        // std::cout << "p1" << std::endl;
        cudaMemset(keys, 0, sizeof(T) * keys_sz);
        // std::cout << "p2" << std::endl;
        return keys;
    }

    void random_shuffle(T *keys, size_t keys_sz) {
        // std::random_shuffle(keys, keys + keys_sz);
    }

    void free(T *ptr) {
        cudaFree(ptr);
    }
};

// template <>
// GPU_Tools<unsigned int>::GPU_Tools(bool random_seed) : CPU_Tools<unsigned int>(random_seed) {
//     CURAND_CALL(curandCreateGenerator(&gen, 
//             CURAND_RNG_PSEUDO_DEFAULT));
//     CURAND_CALL(curandSetGeneratorOffset(gen, this->rand()));
// }

// template <>
// unsigned int * GPU_Tools<unsigned int>::random(unsigned int *keys, size_t keys_sz, 
//     unsigned int min, unsigned int max) {

//     // std::cout << "gpu random, unsigned int, " << keys_sz << std::endl;
//     if (keys == nullptr) {
//         checkKernelErrors(cudaMalloc(&keys, sizeof(unsigned int) * keys_sz));
//     }
//     CURAND_CALL(curandGenerate(gen, keys, keys_sz));
//     cudaDeviceSynchronize();

//     unsigned int *buf = new unsigned int[keys_sz];
//     checkKernelErrors(cudaMemcpy(buf, keys, sizeof(unsigned int) * keys_sz, cudaMemcpyDeviceToHost));

//     // size_t count = 0;

//     // for (size_t i = 0; i < keys_sz; ++i) {
//     //     if (buf[i] == 0) {
//     //         count++;
//     //     }

//     //     // if (keys_sz == 2056) 
//     //     {
//     //         std::cout << "seed: " << buf[i] << std::endl;
//     //     }
//     // }

//     // std::cout << "zero seeds: " << count << std::endl;

//     return keys;
// }
/*
gpu rand end
*/

// template <>
// GPU_Tools<unsigned int>::GPU_Tools(bool random_seed) : CPU_Tools<unsigned int>(random_seed) {
//     CURAND_CALL(curandCreateGenerator(&gen, 
//             CURAND_RNG_QUASI_SOBOL32));
//     CURAND_CALL(curandSetGeneratorOffset(gen, this->rand()));
// }

// template <>
// unsigned int * GPU_Tools<unsigned int>::random(unsigned int *keys, size_t keys_sz, 
//     unsigned int min, unsigned int max) {

//     // std::cout << "gpu random, unsigned int, " << keys_sz << std::endl;
//     if (keys == nullptr) {
//         checkKernelErrors(cudaMalloc(&keys, sizeof(unsigned int) * keys_sz));
//     }
//     CURAND_CALL(curandGenerate(gen, keys, keys_sz));
//     cudaDeviceSynchronize();

//     unsigned int *buf = new unsigned int[keys_sz];
//     checkKernelErrors(cudaMemcpy(buf, keys, sizeof(unsigned int) * keys_sz, cudaMemcpyDeviceToHost));

//     size_t count = 0;

//     for (size_t i = 0; i < keys_sz; ++i) {
//         if (buf[i] == 0) {
//             count++;
//         }

//         // if (keys_sz == 2056) 
//         {
//             std::cout << "seed: " << buf[i] << std::endl;
//         }
//     }

//     // std::cout << "zero seeds: " << count << std::endl;

//     return keys;
// }

// template <>
// GPU_Tools<unsigned long long>::GPU_Tools(bool random_seed) : CPU_Tools<unsigned long long>(random_seed) {
//     CURAND_CALL(curandCreateGenerator(&gen, 
//             CURAND_RNG_QUASI_SOBOL64));
//     CURAND_CALL(curandSetGeneratorOffset(gen, this->rand()));
// }

// template <>
// unsigned long long * GPU_Tools<unsigned long long>::random(unsigned long long *keys, size_t keys_sz, 
//     unsigned long long min, unsigned long long max) {

//     if (keys == nullptr) {
//         checkKernelErrors(cudaMalloc(&keys, sizeof(unsigned long long) * keys_sz));
//     }
//     CURAND_CALL(curandGenerateLongLong(gen, keys, keys_sz));
//     cudaDeviceSynchronize();
//     return keys;
// }

// extern constexpr size_t host_buf_sz;


template <typename Key_T, typename Count_T, typename Hashed_T = size_t, size_t host_buf_sz = 1024 * 1024> // 
struct Unordered_Map_GPU : Unordered_Map<Key_T, Count_T, Hashed_T> {
    // using Unordered_Map<Key_T, Count_T, Hashed_T>::um;
    virtual int insert(Key_T *keys, size_t keys_sz) {

        static Key_T buf[host_buf_sz];

        for (size_t i = 0; i < keys_sz; i += host_buf_sz) {
            // std::cout << i << " p0" << std::endl;
            size_t cpd = std::min(host_buf_sz, keys_sz - i);
            // std::cout << "cpd: " << cpd << std::endl;
            checkKernelErrors(cudaMemcpy(buf, keys + i, sizeof(Key_T) * cpd, cudaMemcpyDeviceToHost));
            // ++um[keys[i]];
            // std::cout << i << " p1" << std::endl;
            Unordered_Map<Key_T, Count_T, Hashed_T>::insert(buf, cpd);
            // std::cout << i << " p2" << std::endl;
        }
        return 0;
    }
    virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
        static Key_T key_buf[host_buf_sz];
        static Count_T count_buf[host_buf_sz];

        for (size_t i = 0; i < keys_sz; i += host_buf_sz) {
            size_t cpd = std::min(host_buf_sz, keys_sz - i);
            checkKernelErrors(cudaMemcpy(key_buf, keys + i, sizeof(Key_T) * cpd, cudaMemcpyDeviceToHost));
            // std::cout << "cpd: " << cpd << std::endl;
            Unordered_Map<Key_T, Count_T, Hashed_T>::search(key_buf + i, cpd, count_buf);
            // for (size_t j = 0; j < cpd; ++j) {
            //     std::cout << count_buf[j] << std::endl;
            // }
            checkKernelErrors(cudaMemcpy(count + i, count_buf, sizeof(Key_T) * cpd, cudaMemcpyHostToDevice));
            // ++um[keys[i]];
        }
        return 0;
    }
};


// template <typename Key_T, typename Count_T, typename Hashed_T = size_t> // 
// struct Binary_Search_GPU {
//     // using Unordered_Map<Key_T, Count_T, Hashed_T>::um;
//     virtual int insert(Key_T *keys, size_t keys_sz) {

//         thrust::sort(keys, keys + keys_sz);
//     }
//     virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
//         Count_T *lb, *ub;
//         // checkKernelErrors(cudaMemcpy(key_buf, keys + i, sizeof(Key_T) * cpd, cudaMemcpyDeviceToHost));
//     }
// };




template <typename T>
GPU_Tools<T> gpu_tool;

extern const size_t default_hash_n;
extern const size_t default_hash_m;
extern const size_t default_hash_nc;
extern const size_t default_hash_nr;


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, 
    typename Hash_Function> // 
struct Count_Min_Sketch_GPU : Sketch<Key_T, Count_T> {

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

    Count_Min_Sketch_GPU(
        Insert_Kernel_Function *_insert_kernel,
        Search_Kernel_Function *_search_kernel,
        size_t _n = default_hash_n, size_t _m = default_hash_m, size_t ss = default_seed_sz) : 
        n(_n), m(_m),
        seed_sz(ss) {

        insert_kernel = _insert_kernel;
        search_kernel = _search_kernel;

        seed_num = 4 * m * WARP_SIZE + 1;
        // std::cout << "seed_num: " << seed_num << std::endl;

        table_total_sz = n * m * WARP_SIZE;
        // std::cout << "table_total_sz: " << table_total_sz << std::endl;
        table = gpu_tool<Count_T>.zero(table, table_total_sz);

        seed_total_sz = seed_sz * seed_num;
        // std::cout << "seed_total_sz: " << seed_total_sz << std::endl;

        // random seeds
        seed = gpu_tool<Seed_T>.random(seed, seed_total_sz, 1, std::numeric_limits<Seed_T>::max());
        // gridDim = dim3(DEFAULT_GRID_DIM_X, 1, 1);
        // blockDim = dim3(DEFAULT_BLOCK_DIM_X, 1, 1);
        gridDim = default_grid_dim;
        blockDim = default_block_dim;

        nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
        nwarps = ceil<size_t>(nthreads, WARP_SIZE);

        // std::cout << "gridDim: " << gridDim << std::endl;
        // std::cout << "blockDim: " << blockDim << std::endl;

        // hash = hash_mul_add<Key_T, Hashed_T, Seed_T>;
        // insert_kernel = insert_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>;

    }

    virtual int insert(Key_T *keys, size_t keys_sz) {
        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        // thrust::device_vector<Hashed_T> debug_vector;
        checkKernelErrors((insert_kernel
            <<<gridDim, blockDim>>>(
                keys, keys_sz,
                table, n, m,
                // Hash_Function<Key_T, Hashed_T, Seed_T>(),
                Hash_Function(),
                seed, seed_sz,
                work_load_per_warp,
                // &debug_vector
                nullptr
            )));
        cudaDeviceSynchronize();
        // std::cout << "GPU insert" << std::endl;

        // {
        //     thrust::host_vector<Hashed_T> hv(debug_vector);
        //     std::unordered_map<Hashed_T, size_t> um;
        //     for (auto &v : hv) {
        //         um[v]++;
        //     }
        //     for (auto &v : um) {
        //         std::cout << v.first << " : " << v.second << std::endl;
        //     }
        // }

        return 0;
    }
    virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        checkKernelErrors((search_kernel
            <<<gridDim, blockDim>>>(
                keys, keys_sz,
                table, n, m,
                Hash_Function(),
                seed, seed_sz,
                count, std::numeric_limits<Count_T>::max(),
                work_load_per_warp,
                nullptr
            )));
        cudaDeviceSynchronize();
        return 0;
    }

    virtual void clear() {
        table = gpu_tool<Count_T>.zero(table, table_total_sz);
    }

    virtual ~Count_Min_Sketch_GPU() {
        cudaFree(table);
        cudaFree(seed);
    }

};

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, 
    typename Hash_Function> // 
struct Count_Min_Sketch_GPU_Levels : Sketch<Key_T, Count_T> {

    using Insert_Kernel_Function = void (Key_T *, size_t,
        Count_T *, size_t , size_t,
        Count_T *, size_t , size_t,
        const Hash_Function &,
        Seed_T *, size_t,
        size_t,
        void *);

    using Search_Kernel_Function = void (Key_T *, size_t,
        Count_T *, size_t , size_t,
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

    size_t n_low;
    size_t m_low;
    size_t table_total_sz_low;
    Count_T *table_low = nullptr;

    size_t n_high;
    size_t m_high;
    size_t table_total_sz_high;
    Count_T *table_high = nullptr;

    size_t seed_sz;
    size_t seed_num;
    size_t seed_total_sz;

    Seed_T *seed = nullptr;

    Count_Min_Sketch_GPU_Levels(
        Insert_Kernel_Function *_insert_kernel,
        Search_Kernel_Function *_search_kernel,
        size_t _n_low = default_hash_n, size_t _m_low = default_hash_m,
        size_t _n_high = default_hash_n, size_t _m_high = default_hash_m,
        size_t ss = default_seed_sz) : 
        // n(_n), m(_m),
        n_low(_n_low), m_low(_m_low), n_high(_n_high), m_high(_m_high),
        seed_sz(ss) {

        insert_kernel = _insert_kernel;
        search_kernel = _search_kernel;

        seed_num = 4 * m_low * WARP_SIZE + 1;
        // std::cout << "seed_num: " << seed_num << std::endl;

        table_total_sz_low = n_low * m_low * WARP_SIZE;
        // std::cout << "table_total_sz: " << table_total_sz << std::endl;
        table_low = gpu_tool<Count_T>.zero(table_low, table_total_sz_low);

        table_total_sz_high = n_high * m_high * WARP_SIZE;
        // std::cout << "table_total_sz: " << table_total_sz << std::endl;
        table_high = gpu_tool<Count_T>.zero(table_high, table_total_sz_high);

        seed_total_sz = seed_sz * seed_num;
        // std::cout << "seed_total_sz: " << seed_total_sz << std::endl;

        // random seeds
        seed = gpu_tool<Seed_T>.random(seed, seed_total_sz, 1, std::numeric_limits<Seed_T>::max());
        // gridDim = dim3(DEFAULT_GRID_DIM_X, 1, 1);
        // blockDim = dim3(DEFAULT_BLOCK_DIM_X, 1, 1);
        gridDim = default_grid_dim;
        blockDim = default_block_dim;

        nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
        nwarps = ceil<size_t>(nthreads, WARP_SIZE);

        // std::cout << "gridDim: " << gridDim << std::endl;
        // std::cout << "blockDim: " << blockDim << std::endl;

        // hash = hash_mul_add<Key_T, Hashed_T, Seed_T>;
        // insert_kernel = insert_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>;

    }

    virtual int insert(Key_T *keys, size_t keys_sz) {
        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        // thrust::device_vector<Hashed_T> debug_vector;
        checkKernelErrors((insert_kernel
            <<<gridDim, blockDim>>>(
                keys, keys_sz,
                table_low, n_low, m_low,
                table_high, n_high, m_high,
                // Hash_Function<Key_T, Hashed_T, Seed_T>(),
                Hash_Function(),
                seed, seed_sz,
                work_load_per_warp,
                // &debug_vector
                nullptr
            )));
        cudaDeviceSynchronize();
        // std::cout << "GPU insert" << std::endl;

        // {
        //     thrust::host_vector<Hashed_T> hv(debug_vector);
        //     std::unordered_map<Hashed_T, size_t> um;
        //     for (auto &v : hv) {
        //         um[v]++;
        //     }
        //     for (auto &v : um) {
        //         std::cout << v.first << " : " << v.second << std::endl;
        //     }
        // }

        return 0;
    }
    virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        checkKernelErrors((search_kernel
            <<<gridDim, blockDim>>>(
                keys, keys_sz,
                table_low, n_low, m_low,
                table_high, n_high, m_high,
                Hash_Function(),
                seed, seed_sz,
                count, std::numeric_limits<Count_T>::max(),
                work_load_per_warp,
                nullptr
            )));
        cudaDeviceSynchronize();
        return 0;
    }

    virtual void clear() {
        table_low = gpu_tool<Count_T>.zero(table_low, table_total_sz_low);
        table_high = gpu_tool<Count_T>.zero(table_high, table_total_sz_high);
    }

    virtual ~Count_Min_Sketch_GPU_Levels() {
        cudaFree(table_low);
        cudaFree(table_high);
        cudaFree(seed);
    }

};

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, 
    typename Hash_Function> // 
struct Count_Min_Sketch_GPU_Mem_Levels : Sketch<Key_T, Count_T> {

    using Insert_Kernel_Function = void (Key_T *, size_t,
        Count_T *, size_t ,
        Count_T *, size_t , unsigned int *,
        const Hash_Function &,
        Seed_T *, size_t,
        size_t,
        void *);

    using Search_Kernel_Function = void (Key_T *, size_t,
        Count_T *, size_t , 
        Count_T *, size_t , unsigned int *,
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

    size_t n_low;
    // size_t m_low;
    size_t table_total_sz_low;
    Count_T *table_low = nullptr;

    size_t n_high;
    // size_t m_high;
    size_t table_total_sz_high;
    Count_T *table_high = nullptr;

    size_t seed_sz;
    size_t seed_num;
    size_t seed_total_sz;

    Seed_T *seed = nullptr;
    // __constant__ Seed_T seed[1024];

    unsigned int *mem_id = nullptr;
    void *debug = nullptr;


    Count_Min_Sketch_GPU_Mem_Levels(
        Insert_Kernel_Function *_insert_kernel,
        Search_Kernel_Function *_search_kernel,
        size_t _n_low = default_hash_n,
        size_t _n_high = default_hash_n,
        size_t ss = default_seed_sz) : 
        // n(_n), m(_m),
        n_low(_n_low), n_high(_n_high), 
        seed_sz(ss) {

        insert_kernel = _insert_kernel;
        search_kernel = _search_kernel;

        seed_num = 4 * WARP_SIZE + 1;
        // std::cout << "seed_num: " << seed_num << std::endl;

        table_total_sz_low = n_low * WARP_SIZE;
        // std::cout << "table_total_sz: " << table_total_sz << std::endl;
        table_low = gpu_tool<Count_T>.zero(table_low, table_total_sz_low);

        table_total_sz_high = n_high * WARP_SIZE;
        // std::cout << "table_total_sz: " << table_total_sz << std::endl;
        table_high = gpu_tool<Count_T>.zero(table_high, table_total_sz_high);

        seed_total_sz = seed_sz * seed_num;
        // std::cout << "seed_total_sz: " << seed_total_sz << std::endl;

        // random seeds
        seed = gpu_tool<Seed_T>.random(seed, seed_total_sz, 1, std::numeric_limits<Seed_T>::max());
        {
//             __constant__ float constData[256];
// float data[256];
// cudaMemcpyToSymbol(constData, data, sizeof(data));
            // cudaMemcpyToSymbol(seed, seed, sizeof(Seed_T) * seed_total_sz);
            copy_to_constant_seed<Seed_T>(seed, seed_total_sz);
        }


        mem_id = gpu_tool<unsigned int>.zero(mem_id, 1);
        {
            unsigned int mem_id_ini_val = BUFFER_START + 1;
            checkKernelErrors(cudaMemcpy(mem_id, &mem_id_ini_val, sizeof(unsigned int), cudaMemcpyHostToDevice));
        }


        debug = gpu_tool<unsigned int>.zero(static_cast<unsigned int *>(debug), 1);

        // gridDim = dim3(DEFAULT_GRID_DIM_X, 1, 1);
        // blockDim = dim3(DEFAULT_BLOCK_DIM_X, 1, 1);
        gridDim = default_grid_dim;
        blockDim = default_block_dim;

        nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
        nwarps = ceil<size_t>(nthreads, WARP_SIZE);

        // std::cout << "gridDim: " << gridDim << std::endl;
        // std::cout << "blockDim: " << blockDim << std::endl;

        // hash = hash_mul_add<Key_T, Hashed_T, Seed_T>;
        // insert_kernel = insert_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>;

    }

    virtual int insert(Key_T *keys, size_t keys_sz) {

        // init_hash_table_high<Count_T><<<gridDim, blockDim>>>(table_high, table_total_sz_high);

        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        // thrust::device_vector<Hashed_T> debug_vector;

        // __constant__ Seed_T seed_c[1024];
        // cudaMemcpyToSymbol(seed_c, seed, sizeof(Seed_T) * seed_total_sz);
        

        checkKernelErrors((insert_kernel
            <<<gridDim, blockDim>>>(
                keys, keys_sz,
                table_low, n_low,
                table_high, n_high, mem_id,
                // Hash_Function<Key_T, Hashed_T, Seed_T>(),
                Hash_Function(),
                seed, seed_sz,
                work_load_per_warp,
                // &debug_vector
                nullptr
            )));
        cudaDeviceSynchronize();

        // unsigned int mem_id_host;
        // checkKernelErrors(cudaMemcpy(&mem_id_host, mem_id, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // std::cout << "mem_id_host: " << mem_id_host << std::endl;
        // std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;


        return 0;
    }
    virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        checkKernelErrors((search_kernel
            <<<gridDim, blockDim>>>(
                keys, keys_sz,
                table_low, n_low,
                table_high, n_high, mem_id,
                Hash_Function(),
                seed, seed_sz,
                count, std::numeric_limits<Count_T>::max(),
                work_load_per_warp,
                debug
            )));
        cudaDeviceSynchronize();

        // unsigned int debug_host;
        // checkKernelErrors(cudaMemcpy(&debug_host, debug, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // std::cout << "debug_host: " << debug_host << std::endl;
        // std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;
        return 0;
    }

    virtual void clear() {
        table_low = gpu_tool<Count_T>.zero(table_low, table_total_sz_low);
        table_high = gpu_tool<Count_T>.zero(table_high, table_total_sz_high);
        mem_id = gpu_tool<unsigned int>.zero(mem_id, 1);
        {
            unsigned int mem_id_ini_val = BUFFER_START + 1;
            checkKernelErrors(cudaMemcpy(mem_id, &mem_id_ini_val, sizeof(unsigned int), cudaMemcpyHostToDevice));
        }
        cudaDeviceSynchronize();
    }

    virtual ~Count_Min_Sketch_GPU_Mem_Levels() {
        cudaFree(table_low);
        cudaFree(table_high);
        cudaFree(seed);
        cudaFree(mem_id);
        cudaFree(debug);
    }

};


template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, 
    typename Hash_Function> // 
struct Count_Min_Sketch_GPU_Mem_Levels_Pre_Cal : Sketch<Key_T, Count_T> {

    using Insert_Kernel_Function = void (Key_T *, size_t,
        Count_T *, size_t ,
        Count_T *, size_t , unsigned int *,
        const Hash_Function &,
        Seed_T *, size_t,
        unsigned char *,
        size_t,
        void *);

    using Search_Kernel_Function = void (Key_T *, size_t,
        Count_T *, size_t , 
        Count_T *, size_t , unsigned int *,
        const Hash_Function &,
        Seed_T *, size_t,
        unsigned char *,
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

    size_t n_low;
    // size_t m_low;
    size_t table_total_sz_low;
    Count_T *table_low = nullptr;

    size_t n_high;
    // size_t m_high;
    size_t table_total_sz_high;
    Count_T *table_high = nullptr;

    size_t seed_sz;
    size_t seed_num;
    size_t seed_total_sz;

    Seed_T *seed = nullptr;
    // __constant__ Seed_T seed[1024];

    unsigned int *mem_id = nullptr;
    unsigned char *hash_mask_table = nullptr;



    void *debug = nullptr;


    Count_Min_Sketch_GPU_Mem_Levels_Pre_Cal(
        Insert_Kernel_Function *_insert_kernel,
        Search_Kernel_Function *_search_kernel,
        size_t _n_low = default_hash_n,
        size_t _n_high = default_hash_n,
        size_t ss = default_seed_sz) : 
        // n(_n), m(_m),
        n_low(_n_low), n_high(_n_high), 
        seed_sz(ss) {

        insert_kernel = _insert_kernel;
        search_kernel = _search_kernel;

        seed_num = 4 * WARP_SIZE + 1;
        // std::cout << "seed_num: " << seed_num << std::endl;

        table_total_sz_low = n_low * WARP_SIZE;
        // std::cout << "table_total_sz: " << table_total_sz << std::endl;
        table_low = gpu_tool<Count_T>.zero(table_low, table_total_sz_low);

        table_total_sz_high = n_high * WARP_SIZE;
        // std::cout << "table_total_sz: " << table_total_sz << std::endl;
        table_high = gpu_tool<Count_T>.zero(table_high, table_total_sz_high);

        seed_total_sz = seed_sz * seed_num;
        // std::cout << "seed_total_sz: " << seed_total_sz << std::endl;

        // random seeds
        seed = gpu_tool<Seed_T>.random(seed, seed_total_sz, 1, std::numeric_limits<Seed_T>::max());
        {
//             __constant__ float constData[256];
// float data[256];
// cudaMemcpyToSymbol(constData, data, sizeof(data));
            // cudaMemcpyToSymbol(seed, seed, sizeof(Seed_T) * seed_total_sz);
            copy_to_constant_seed<Seed_T>(seed, seed_total_sz);
        }


        mem_id = gpu_tool<unsigned int>.zero(mem_id, 1);
        {
            unsigned int mem_id_ini_val = BUFFER_START + 1;
            checkKernelErrors(cudaMemcpy(mem_id, &mem_id_ini_val, sizeof(unsigned int), cudaMemcpyHostToDevice));
        }


        debug = gpu_tool<unsigned int>.zero(static_cast<unsigned int *>(debug), 1);

        // gridDim = dim3(DEFAULT_GRID_DIM_X, 1, 1);
        // blockDim = dim3(DEFAULT_BLOCK_DIM_X, 1, 1);
        gridDim = default_grid_dim;
        blockDim = default_block_dim;

        nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
        nwarps = ceil<size_t>(nthreads, WARP_SIZE);

        // std::cout << "gridDim: " << gridDim << std::endl;
        // std::cout << "blockDim: " << blockDim << std::endl;

        // hash = hash_mul_add<Key_T, Hashed_T, Seed_T>;
        // insert_kernel = insert_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>;

    }

    virtual unsigned char *pre_cal(Key_T *keys, size_t keys_sz, void *debug) {
        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        // thrust::device_vector<Hashed_T> debug_vector;

        // __constant__ Seed_T seed_c[1024];
        // cudaMemcpyToSymbol(seed_c, seed, sizeof(Seed_T) * seed_total_sz);

        // keys_sz = 1024;
        // std::cout << "hellw world 1" << std::endl;
        if (hash_mask_table != nullptr) {
            cudaFree(hash_mask_table);
            hash_mask_table = nullptr;
        }

        hash_mask_table = gpu_tool<unsigned char>.zero(hash_mask_table, keys_sz * 16);
        // Count_T *debug_count = gpu_tool<Count_T>.zero(nullptr, 1);
        
        
        checkKernelErrors((pre_cal_kernel<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>
            <<<gridDim, blockDim>>>(
                keys, keys_sz,
                hash_mask_table,
                // Hash_Function<Key_T, Hashed_T, Seed_T>(),
                Hash_Function(),
                seed, seed_sz,
                work_load_per_warp,
                // &debug_vector
                // nullptr
                debug
                // debug_count
            )));
        cudaDeviceSynchronize();
        // std::cout << "hellw world 2" << std::endl;

        // Count_T hdebug_count;
        // checkKernelErrors(cudaMemcpy(&hdebug_count, debug_count, sizeof(Count_T), cudaMemcpyDeviceToHost));
        // std::cout << "hdebug_count: " << hdebug_count << std::endl;

        return hash_mask_table;
    }

    void free_hash_mask_table() {
        cudaFree(hash_mask_table);
        hash_mask_table = nullptr;
    }



    virtual int insert(Key_T *keys, size_t keys_sz) {

        // init_hash_table_high<Count_T><<<gridDim, blockDim>>>(table_high, table_total_sz_high);

        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        // thrust::device_vector<Hashed_T> debug_vector;

        // __constant__ Seed_T seed_c[1024];
        // cudaMemcpyToSymbol(seed_c, seed, sizeof(Seed_T) * seed_total_sz);
        

        checkKernelErrors((insert_kernel
            <<<gridDim, blockDim>>>(
                keys, keys_sz,
                table_low, n_low,
                table_high, n_high, mem_id,
                // Hash_Function<Key_T, Hashed_T, Seed_T>(),
                Hash_Function(),
                seed, seed_sz,
                hash_mask_table,
                work_load_per_warp,
                // &debug_vector
                nullptr
            )));
        cudaDeviceSynchronize();

        // unsigned int mem_id_host;
        // checkKernelErrors(cudaMemcpy(&mem_id_host, mem_id, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // std::cout << "mem_id_host: " << mem_id_host << std::endl;
        // std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;


        return 0;
    }
    virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        checkKernelErrors((search_kernel
            <<<gridDim, blockDim>>>(
                keys, keys_sz,
                table_low, n_low,
                table_high, n_high, mem_id,
                Hash_Function(),
                seed, seed_sz,
                hash_mask_table,
                count, std::numeric_limits<Count_T>::max(),
                work_load_per_warp,
                debug
            )));
        cudaDeviceSynchronize();

        // unsigned int debug_host;
        // checkKernelErrors(cudaMemcpy(&debug_host, debug, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // std::cout << "debug_host: " << debug_host << std::endl;
        // std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;
        return 0;
    }

    virtual void clear() {
        table_low = gpu_tool<Count_T>.zero(table_low, table_total_sz_low);
        table_high = gpu_tool<Count_T>.zero(table_high, table_total_sz_high);
        mem_id = gpu_tool<unsigned int>.zero(mem_id, 1);
        {
            unsigned int mem_id_ini_val = BUFFER_START + 1;
            checkKernelErrors(cudaMemcpy(mem_id, &mem_id_ini_val, sizeof(unsigned int), cudaMemcpyHostToDevice));
        }

        // cudaFree(hash_mask_table);
        // hash_mask_table = nullptr;
        cudaDeviceSynchronize();

    }

    virtual ~Count_Min_Sketch_GPU_Mem_Levels_Pre_Cal() {
        cudaFree(table_low);
        cudaFree(table_high);
        cudaFree(seed);
        cudaFree(mem_id);
        cudaFree(hash_mask_table);
        cudaFree(debug);
    }

};

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, 
    typename Hash_Function> // 
struct Count_Min_Sketch_GPU_Mem_Levels_Host_Pre_Cal : Sketch<Key_T, Count_T> {

    using Insert_Kernel_Function = void (Key_T *, size_t,
        Count_T *, size_t ,
        Count_T *, size_t , unsigned int *,
        const Hash_Function &,
        Seed_T *, size_t,
        unsigned char *,
        size_t,
        void *);

    using Search_Kernel_Function = void (Key_T *, size_t,
        Count_T *, size_t , 
        Count_T *, size_t , unsigned int *,
        const Hash_Function &,
        Seed_T *, size_t,
        unsigned char *,
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

    size_t n_low;
    // size_t m_low;
    size_t table_total_sz_low;
    Count_T *table_low = nullptr;

    size_t n_high;
    // size_t m_high;
    size_t table_total_sz_high;
    Count_T *table_high = nullptr;

    size_t seed_sz;
    size_t seed_num;
    size_t seed_total_sz;

    Seed_T *seed = nullptr;
    // __constant__ Seed_T seed[1024];

    unsigned int *mem_id = nullptr;
    unsigned char *hash_mask_table = nullptr;



    void *debug = nullptr;


    Count_Min_Sketch_GPU_Mem_Levels_Host_Pre_Cal(
        Insert_Kernel_Function *_insert_kernel,
        Search_Kernel_Function *_search_kernel,
        size_t _n_low = default_hash_n,
        size_t _n_high = default_hash_n,
        size_t ss = default_seed_sz) : 
        // n(_n), m(_m),
        n_low(_n_low), n_high(_n_high), 
        seed_sz(ss) {

        insert_kernel = _insert_kernel;
        search_kernel = _search_kernel;

        seed_num = 4 * WARP_SIZE + 1;
        // std::cout << "seed_num: " << seed_num << std::endl;

        table_total_sz_low = n_low * WARP_SIZE;
        // std::cout << "table_total_sz: " << table_total_sz << std::endl;
        table_low = gpu_tool<Count_T>.zero(table_low, table_total_sz_low);

        table_total_sz_high = n_high * WARP_SIZE;
        // std::cout << "table_total_sz: " << table_total_sz << std::endl;
        table_high = gpu_tool<Count_T>.zero(table_high, table_total_sz_high);

        seed_total_sz = seed_sz * seed_num;
        // std::cout << "seed_total_sz: " << seed_total_sz << std::endl;

        // random seeds
        seed = gpu_tool<Seed_T>.random(seed, seed_total_sz, 1, std::numeric_limits<Seed_T>::max());
        {
//             __constant__ float constData[256];
// float data[256];
// cudaMemcpyToSymbol(constData, data, sizeof(data));
            // cudaMemcpyToSymbol(seed, seed, sizeof(Seed_T) * seed_total_sz);
            copy_to_constant_seed<Seed_T>(seed, seed_total_sz);
        }


        mem_id = gpu_tool<unsigned int>.zero(mem_id, 1);
        {
            unsigned int mem_id_ini_val = BUFFER_START + 1;
            checkKernelErrors(cudaMemcpy(mem_id, &mem_id_ini_val, sizeof(unsigned int), cudaMemcpyHostToDevice));
        }


        debug = gpu_tool<unsigned int>.zero(static_cast<unsigned int *>(debug), 1);

        // gridDim = dim3(DEFAULT_GRID_DIM_X, 1, 1);
        // blockDim = dim3(DEFAULT_BLOCK_DIM_X, 1, 1);
        gridDim = default_grid_dim;
        blockDim = default_block_dim;

        nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
        nwarps = ceil<size_t>(nthreads, WARP_SIZE);

        // std::cout << "gridDim: " << gridDim << std::endl;
        // std::cout << "blockDim: " << blockDim << std::endl;

        // hash = hash_mul_add<Key_T, Hashed_T, Seed_T>;
        // insert_kernel = insert_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>;

    }

    virtual unsigned char *pre_cal(Key_T *keys, size_t keys_sz, void *debug) {
        
        if (hash_mask_table != nullptr) {
            cudaFree(hash_mask_table);
            hash_mask_table = nullptr;
        }

        hash_mask_table = gpu_tool<unsigned char>.zero(hash_mask_table, keys_sz * 16);

        unsigned char *hash_mask_table_host = nullptr;
        hash_mask_table_host = cpu_tool<unsigned char>.zero(hash_mask_table_host, keys_sz * 16);

        pre_cal_host(hash_mask_table_host, keys_sz, HASH_MASK_SIZE);  
        
        checkKernelErrors(cudaMemcpy(hash_mask_table, hash_mask_table_host, keys_sz * HASH_MASK_SIZE, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        // std::cout << "hellw world 2" << std::endl;
        // Count_T *debug_count = gpu_tool<Count_T>.zero(nullptr, 1);
        // Count_T hdebug_count;
        // checkKernelErrors(cudaMemcpy(&hdebug_count, debug_count, sizeof(Count_T), cudaMemcpyDeviceToHost));
        // std::cout << "hdebug_count: " << hdebug_count << std::endl;

        return hash_mask_table;
    }

    void free_hash_mask_table() {
        cudaFree(hash_mask_table);
        hash_mask_table = nullptr;
    }



    virtual int insert(Key_T *keys, size_t keys_sz) {

        // init_hash_table_high<Count_T><<<gridDim, blockDim>>>(table_high, table_total_sz_high);

        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        // thrust::device_vector<Hashed_T> debug_vector;

        // __constant__ Seed_T seed_c[1024];
        // cudaMemcpyToSymbol(seed_c, seed, sizeof(Seed_T) * seed_total_sz);
        

        checkKernelErrors((insert_kernel
            <<<gridDim, blockDim>>>(
                keys, keys_sz,
                table_low, n_low,
                table_high, n_high, mem_id,
                // Hash_Function<Key_T, Hashed_T, Seed_T>(),
                Hash_Function(),
                seed, seed_sz,
                hash_mask_table,
                work_load_per_warp,
                // &debug_vector
                nullptr
            )));
        cudaDeviceSynchronize();

        // unsigned int mem_id_host;
        // checkKernelErrors(cudaMemcpy(&mem_id_host, mem_id, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // std::cout << "mem_id_host: " << mem_id_host << std::endl;
        // std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;


        return 0;
    }
    virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        checkKernelErrors((search_kernel
            <<<gridDim, blockDim>>>(
                keys, keys_sz,
                table_low, n_low,
                table_high, n_high, mem_id,
                Hash_Function(),
                seed, seed_sz,
                hash_mask_table,
                count, std::numeric_limits<Count_T>::max(),
                work_load_per_warp,
                debug
            )));
        cudaDeviceSynchronize();

        // unsigned int debug_host;
        // checkKernelErrors(cudaMemcpy(&debug_host, debug, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // std::cout << "debug_host: " << debug_host << std::endl;
        // std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;
        return 0;
    }

    virtual void clear() {
        table_low = gpu_tool<Count_T>.zero(table_low, table_total_sz_low);
        table_high = gpu_tool<Count_T>.zero(table_high, table_total_sz_high);
        mem_id = gpu_tool<unsigned int>.zero(mem_id, 1);
        {
            unsigned int mem_id_ini_val = BUFFER_START + 1;
            checkKernelErrors(cudaMemcpy(mem_id, &mem_id_ini_val, sizeof(unsigned int), cudaMemcpyHostToDevice));
        }

        // cudaFree(hash_mask_table);
        // hash_mask_table = nullptr;
        cudaDeviceSynchronize();

    }

    virtual ~Count_Min_Sketch_GPU_Mem_Levels_Host_Pre_Cal() {
        cudaFree(table_low);
        cudaFree(table_high);
        cudaFree(seed);
        cudaFree(mem_id);
        cudaFree(hash_mask_table);
        cudaFree(debug);
    }

};

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, 
    typename Hash_Function> // 
struct Count_Min_Sketch_GPU_Mem_Levels_Host_Pre_Cal_Sub_Warp : Sketch<Key_T, Count_T> {

    using Insert_Kernel_Function = void (Key_T *, size_t,
        Count_T *, size_t ,
        Count_T *, size_t , unsigned int *,
        const Hash_Function &,
        Seed_T *, size_t,
        unsigned char *,
        size_t,
        void *);

    using Search_Kernel_Function = void (Key_T *, size_t,
        Count_T *, size_t , 
        Count_T *, size_t , unsigned int *,
        const Hash_Function &,
        Seed_T *, size_t,
        unsigned char *,
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

    size_t n_low;
    // size_t m_low;
    size_t table_total_sz_low;
    Count_T *table_low = nullptr;

    size_t n_high;
    // size_t m_high;
    size_t table_total_sz_high;
    Count_T *table_high = nullptr;

    size_t seed_sz;
    size_t seed_num;
    size_t seed_total_sz;

    Seed_T *seed = nullptr;
    // __constant__ Seed_T seed[1024];

    unsigned int *mem_id = nullptr;
    unsigned char *hash_mask_table = nullptr;



    void *debug = nullptr;
    size_t *debug_insert;
    size_t *debug_search;


    Count_Min_Sketch_GPU_Mem_Levels_Host_Pre_Cal_Sub_Warp(
        Insert_Kernel_Function *_insert_kernel,
        Search_Kernel_Function *_search_kernel,
        size_t _n_low = default_hash_n,
        size_t _n_high = default_hash_n,
        size_t ss = default_seed_sz) : 
        // n(_n), m(_m),
        n_low(_n_low), n_high(_n_high), 
        seed_sz(ss) {

        insert_kernel = _insert_kernel;
        search_kernel = _search_kernel;

        seed_num = 4 * WARP_SIZE + 1;
        // std::cout << "seed_num: " << seed_num << std::endl;

        table_total_sz_low = n_low * SUB_WARP_SIZE;
        // std::cout << "table_total_sz: " << table_total_sz << std::endl;
        table_low = gpu_tool<Count_T>.zero(table_low, table_total_sz_low);

        table_total_sz_high = n_high * WARP_SIZE;
        // std::cout << "table_total_sz: " << table_total_sz << std::endl;
        table_high = gpu_tool<Count_T>.zero(table_high, table_total_sz_high);

        seed_total_sz = seed_sz * seed_num;
        // std::cout << "seed_total_sz: " << seed_total_sz << std::endl;

        // random seeds
        seed = gpu_tool<Seed_T>.random(seed, seed_total_sz, 1, std::numeric_limits<Seed_T>::max());
        {
//             __constant__ float constData[256];
// float data[256];
// cudaMemcpyToSymbol(constData, data, sizeof(data));
            // cudaMemcpyToSymbol(seed, seed, sizeof(Seed_T) * seed_total_sz);
            copy_to_constant_seed<Seed_T>(seed, seed_total_sz);
        }


        mem_id = gpu_tool<unsigned int>.zero(mem_id, 1);
        {
            unsigned int mem_id_ini_val = BUFFER_START + 1;
            checkKernelErrors(cudaMemcpy(mem_id, &mem_id_ini_val, sizeof(unsigned int), cudaMemcpyHostToDevice));
        }


        // debug = gpu_tool<unsigned int>.zero(static_cast<unsigned int *>(debug), 1);
        debug_insert = gpu_tool<size_t>.zero(nullptr, 1024 * 1024);
        debug_search = gpu_tool<size_t>.zero(nullptr, 1024 * 1024);

        // gridDim = dim3(DEFAULT_GRID_DIM_X, 1, 1);
        // blockDim = dim3(DEFAULT_BLOCK_DIM_X, 1, 1);
        gridDim = default_grid_dim;
        blockDim = default_block_dim;

        nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
        nwarps = ceil<size_t>(nthreads, WARP_SIZE);

        // std::cout << "gridDim: " << gridDim << std::endl;
        // std::cout << "blockDim: " << blockDim << std::endl;

        // hash = hash_mul_add<Key_T, Hashed_T, Seed_T>;
        // insert_kernel = insert_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>;

    }

    virtual unsigned char *pre_cal(Key_T *keys, size_t keys_sz, void *debug) {
        
        // std::cout << "keys_sz: " << keys_sz << std::endl;

        if (hash_mask_table != nullptr) {
            cudaFree(hash_mask_table);
            hash_mask_table = nullptr;
        }

        hash_mask_table = gpu_tool<unsigned char>.zero(hash_mask_table, keys_sz * HASH_MASK_SIZE_SUB_WARP);

        unsigned char *hash_mask_table_host = nullptr;
        hash_mask_table_host = cpu_tool<unsigned char>.zero(hash_mask_table_host, keys_sz * HASH_MASK_SIZE_SUB_WARP);

        // pre_cal_host(nullptr, keys_sz, HASH_MASK_SIZE);  
        pre_cal_host_sub_warp(hash_mask_table_host, keys_sz, HASH_MASK_SIZE_SUB_WARP);
        // pre_cal_host_sub_warp(nullptr, keys_sz, HASH_MASK_SIZE_SUB_WARP);
        
        checkKernelErrors(cudaMemcpy(hash_mask_table, hash_mask_table_host, keys_sz * HASH_MASK_SIZE_SUB_WARP, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        // std::cout << "hellw world 2" << std::endl;
        // Count_T *debug_count = gpu_tool<Count_T>.zero(nullptr, 1);
        // std::cout << "hellw world 3" << std::endl;
        // Count_T hdebug_count;
        // checkKernelErrors(cudaMemcpy(&hdebug_count, debug_count, sizeof(Count_T), cudaMemcpyDeviceToHost));
        // std::cout << "hdebug_count: " << hdebug_count << std::endl;
        // cudaDeviceSynchronize();

        // std::cout.write((const char *)hash_mask_table_host, keys_sz * HASH_MASK_SIZE_SUB_WARP);

        // for (size_t j = 0; j < keys_sz * HASH_MASK_SIZE_SUB_WARP; ++j) {
        //     std::cout << int(hash_mask_table_host[j]) << std::endl;
        // }

        return hash_mask_table;
    }

    void free_hash_mask_table() {
        cudaFree(hash_mask_table);
        hash_mask_table = nullptr;
    }



    virtual int insert(Key_T *keys, size_t keys_sz) {

        // init_hash_table_high<Count_T><<<gridDim, blockDim>>>(table_high, table_total_sz_high);

        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);

        // std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;
        // thrust::device_vector<Hashed_T> debug_vector;

        // __constant__ Seed_T seed_c[1024];
        // cudaMemcpyToSymbol(seed_c, seed, sizeof(Seed_T) * seed_total_sz);
        

        checkKernelErrors((insert_kernel
            <<<gridDim, blockDim>>>(
                keys, keys_sz,
                table_low, n_low,
                table_high, n_high, mem_id,
                // Hash_Function<Key_T, Hashed_T, Seed_T>(),
                Hash_Function(),
                seed, seed_sz,
                hash_mask_table,
                work_load_per_warp,
                // &debug_vector
                // nullptr
                debug_insert
            )));
        cudaDeviceSynchronize();

        // unsigned int mem_id_host;
        // checkKernelErrors(cudaMemcpy(&mem_id_host, mem_id, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // std::cout << "mem_id_host: " << mem_id_host << std::endl;
        // std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;

        // {
        //     Count_T *h_table_low = new Count_T[table_total_sz_low];
        //     checkKernelErrors(cudaMemcpy(h_table_low, table_low, table_total_sz_low * sizeof(Count_T), cudaMemcpyDeviceToHost));


        //     // for (size_t j = 0; j < table_total_sz_low; ++j) {
        //     //     std::cout << h_table_low[j] << std::endl;
        //     // }

        //     // unsigned char *p = (unsigned char *)h_table_low;
        //     // for (size_t j = 0; j < table_total_sz_low * sizeof(Count_T); ++j) {
        //     //     if (p[j] >= 2)
        //     //         std::cout << int(p[j]) << std::endl;
        //     // }
        // }


        return 0;
    }
    virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        // std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;

        // std::cout << "check count" << std::endl;
        // // for (size_t i = 0; i < keys_sz; ++i) {
        // //     std::cout << i << std::endl;
        // //     atomicMin(count + i, 100);
        // // }

        // check_count<Count_T><<<1, 32>>>(count, keys_sz);
        // std::cout << "check count end" << std::endl;


        checkKernelErrors((search_kernel
            <<<gridDim, blockDim>>>(
                keys, keys_sz,
                table_low, n_low,
                table_high, n_high, mem_id,
                Hash_Function(),
                seed, seed_sz,
                hash_mask_table,
                count, std::numeric_limits<Count_T>::max(),
                work_load_per_warp,
                // debug
                debug_search
            )));
        cudaDeviceSynchronize();

        // unsigned int debug_host;
        // checkKernelErrors(cudaMemcpy(&debug_host, debug, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // std::cout << "debug_host: " << debug_host << std::endl;
        // std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;

        // auto dc3 = diff(thrust::device_vector<size_t>(debug_insert, debug_insert + 1024 * 1024), 
        //         thrust::device_vector<size_t>(debug_search, debug_search + 1024 * 1024));
        // auto max = max_diff(dc3);

        // std::cout << "hash_mask_id_diff: " << max << std::endl;

        return 0;
    }

    virtual void clear() {
        table_low = gpu_tool<Count_T>.zero(table_low, table_total_sz_low);
        table_high = gpu_tool<Count_T>.zero(table_high, table_total_sz_high);
        mem_id = gpu_tool<unsigned int>.zero(mem_id, 1);
        {
            unsigned int mem_id_ini_val = BUFFER_START + 1;
            checkKernelErrors(cudaMemcpy(mem_id, &mem_id_ini_val, sizeof(unsigned int), cudaMemcpyHostToDevice));
        }

        // cudaFree(hash_mask_table);
        // hash_mask_table = nullptr;
        cudaDeviceSynchronize();

    }

    virtual ~Count_Min_Sketch_GPU_Mem_Levels_Host_Pre_Cal_Sub_Warp() {
        cudaFree(table_low);
        cudaFree(table_high);
        cudaFree(seed);
        cudaFree(mem_id);
        cudaFree(hash_mask_table);
        cudaFree(debug);
    }

};

// template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, 
//     typename Hash_Function> // 
// struct Count_Min_Sketch_GPU_Mem_Levels_Bus_Width : Sketch<Key_T, Count_T> {

//     using Insert_Kernel_Function = void (Key_T *, size_t,
//         Count_T *, size_t ,
//         Count_T *, size_t , unsigned int *,
//         const Hash_Function &,
//         Seed_T *, size_t,
//         size_t,
//         void *);

//     using Search_Kernel_Function = void (Key_T *, size_t,
//         Count_T *, size_t , 
//         Count_T *, size_t , unsigned int *,
//         const Hash_Function &,
//         Seed_T *, size_t,
//         Count_T *, Count_T,
//         size_t,
//         void *);

//     dim3 gridDim;
//     dim3 blockDim;

//     size_t nwarps;
//     size_t nthreads;

//     // Hash_Function *hash = nullptr;
//     Insert_Kernel_Function *insert_kernel = nullptr;
//     Search_Kernel_Function *search_kernel = nullptr;

//     size_t n_low;
//     // size_t m_low;
//     size_t table_total_sz_low;
//     Count_T *table_low = nullptr;

//     size_t n_high;
//     // size_t m_high;
//     size_t table_total_sz_high;
//     Count_T *table_high = nullptr;

//     size_t seed_sz;
//     size_t seed_num;
//     size_t seed_total_sz;

//     Seed_T *seed = nullptr;
//     // __constant__ Seed_T seed[1024];

//     unsigned int *mem_id = nullptr;
//     void *debug = nullptr;


//     Count_Min_Sketch_GPU_Mem_Levels(
//         Insert_Kernel_Function *_insert_kernel,
//         Search_Kernel_Function *_search_kernel,
//         size_t _n_low = default_hash_n,
//         size_t _n_high = default_hash_n,
//         size_t ss = default_seed_sz) : 
//         // n(_n), m(_m),
//         n_low(_n_low), n_high(_n_high), 
//         seed_sz(ss) {

//         insert_kernel = _insert_kernel;
//         search_kernel = _search_kernel;

//         seed_num = 4 * WARP_SIZE + 1;
//         // std::cout << "seed_num: " << seed_num << std::endl;

//         table_total_sz_low = n_low * BUS_WIDTH;
//         // std::cout << "table_total_sz: " << table_total_sz << std::endl;
//         table_low = gpu_tool<Count_T>.zero(table_low, table_total_sz_low);

//         table_total_sz_high = n_high * BUS_WIDTH;
//         // std::cout << "table_total_sz: " << table_total_sz << std::endl;
//         table_high = gpu_tool<Count_T>.zero(table_high, table_total_sz_high);

//         seed_total_sz = seed_sz * seed_num;
//         std::cout << "seed_total_sz: " << seed_total_sz << std::endl;

//         // random seeds
//         seed = gpu_tool<Seed_T>.random(seed, seed_total_sz, 1, std::numeric_limits<Seed_T>::max());
//         {
// //             __constant__ float constData[256];
// // float data[256];
// // cudaMemcpyToSymbol(constData, data, sizeof(data));
//             // cudaMemcpyToSymbol(seed, seed, sizeof(Seed_T) * seed_total_sz);
//             copy_to_constant_seed<Seed_T>(seed, seed_total_sz);
//         }


//         mem_id = gpu_tool<unsigned int>.zero(mem_id, 1);
//         {
//             unsigned int mem_id_ini_val = BUFFER_START + 1;
//             checkKernelErrors(cudaMemcpy(mem_id, &mem_id_ini_val, sizeof(unsigned int), cudaMemcpyHostToDevice));
//         }


//         debug = gpu_tool<unsigned int>.zero(static_cast<unsigned int *>(debug), 1);

//         // gridDim = dim3(DEFAULT_GRID_DIM_X, 1, 1);
//         // blockDim = dim3(DEFAULT_BLOCK_DIM_X, 1, 1);
//         gridDim = default_grid_dim;
//         blockDim = default_block_dim;

//         nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
//         nwarps = ceil<size_t>(nthreads, WARP_SIZE);

//         // std::cout << "gridDim: " << gridDim << std::endl;
//         // std::cout << "blockDim: " << blockDim << std::endl;

//         // hash = hash_mul_add<Key_T, Hashed_T, Seed_T>;
//         // insert_kernel = insert_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>;

//     }

//     virtual int insert(Key_T *keys, size_t keys_sz) {

//         // init_hash_table_high<Count_T><<<gridDim, blockDim>>>(table_high, table_total_sz_high);

//         size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
//         // thrust::device_vector<Hashed_T> debug_vector;

//         // __constant__ Seed_T seed_c[1024];
//         // cudaMemcpyToSymbol(seed_c, seed, sizeof(Seed_T) * seed_total_sz);
        

//         checkKernelErrors((insert_kernel
//             <<<gridDim, blockDim>>>(
//                 keys, keys_sz,
//                 table_low, n_low,
//                 table_high, n_high, mem_id,
//                 // Hash_Function<Key_T, Hashed_T, Seed_T>(),
//                 Hash_Function(),
//                 seed, seed_sz,
//                 work_load_per_warp,
//                 // &debug_vector
//                 nullptr
//             )));
//         cudaDeviceSynchronize();

//         unsigned int mem_id_host;
//         checkKernelErrors(cudaMemcpy(&mem_id_host, mem_id, sizeof(unsigned int), cudaMemcpyDeviceToHost));
//         // std::cout << "mem_id_host: " << mem_id_host << std::endl;
//         // std::cout << mem_id_host << std::endl;
//         std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;


//         return 0;
//     }
//     virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
//         size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
//         checkKernelErrors((search_kernel
//             <<<gridDim, blockDim>>>(
//                 keys, keys_sz,
//                 table_low, n_low,
//                 table_high, n_high, mem_id,
//                 Hash_Function(),
//                 seed, seed_sz,
//                 count, std::numeric_limits<Count_T>::max(),
//                 work_load_per_warp,
//                 debug
//             )));
//         cudaDeviceSynchronize();

//         // unsigned int debug_host;
//         // checkKernelErrors(cudaMemcpy(&debug_host, debug, sizeof(unsigned int), cudaMemcpyDeviceToHost));
//         // std::cout << "debug_host: " << debug_host << std::endl;
//         // std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;
//         return 0;
//     }

//     virtual void clear() {
//         table_low = gpu_tool<Count_T>.zero(table_low, table_total_sz_low);
//         table_high = gpu_tool<Count_T>.zero(table_high, table_total_sz_high);
//         mem_id = gpu_tool<unsigned int>.zero(mem_id, 1);
//         {
//             unsigned int mem_id_ini_val = BUFFER_START + 1;
//             checkKernelErrors(cudaMemcpy(mem_id, &mem_id_ini_val, sizeof(unsigned int), cudaMemcpyHostToDevice));
//         }
//     }

//     virtual ~Count_Min_Sketch_GPU_Mem_Levels() {
//         cudaFree(table_low);
//         cudaFree(table_high);
//         cudaFree(seed);
//         cudaFree(mem_id);
//         cudaFree(debug);
//     }

// };


// template <typename Key_T, typename Count_T>
// struct Buffer_Sketch : Sketch<Key_T, Count_T> {

//     Buffer_Sketch(size_t buf_num, size_t buf_sz) {

//     }

//     int insert(const Key_T &key, const Count_T &count) {

//     }

//     Count_T search(const Key_T &key) {

//     }

//     virtual int insert(Key_T *keys, size_t keys_sz) override {
//         for (size_t i = 0; i < keys_sz; ++i) {
//             insert(keys[i], 1);
//         }
//         return 0;
//     }
// };
// template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, 
//     typename Hash_Function> // 
// struct Count_Min_Sketch_GPU_Buffer : Sketch<Key_T, Count_T> {

//     using Insert_Kernel_Function = void (Key_T *, size_t,
//         Count_T *, size_t , size_t,
//         const Hash_Function &,
//         Seed_T *, size_t,
//         size_t,
//         void *);

//     using Search_Kernel_Function = void (Key_T *, size_t,
//         Count_T *, size_t , size_t,
//         const Hash_Function &,
//         Seed_T *, size_t,
//         Count_T *, Count_T,
//         size_t,
//         void *);

//     dim3 gridDim;
//     dim3 blockDim;

//     size_t nwarps;
//     size_t nthreads;

//     // Hash_Function *hash = nullptr;
//     Insert_Kernel_Function *insert_kernel = nullptr;
//     Search_Kernel_Function *search_kernel = nullptr;

//     size_t n;
//     size_t m;
//     size_t table_total_sz;
//     Count_T *table = nullptr;

//     size_t nc;
//     size_t nr;
//     // size_t nl;
//     size_t buffer_total_sz;
//     Key_T *buffer = nullptr;

//     size_t seed_sz;
//     size_t seed_num;
//     size_t seed_total_sz;

//     Seed_T *seed = nullptr;

//     // stream
//     cudaStream_t buffer_steam;
//     cudaStream_t worker_steam;

//     Count_Min_Sketch_GPU(
//         Insert_Kernel_Function *_insert_kernel,
//         Search_Kernel_Function *_search_kernel,
//         size_t _n = default_hash_n, size_t _m = default_hash_m, 
//         size_t _nc = default_hash_nc, size_t _nr = default_hash_nr,
//         size_t ss = default_seed_sz) : 
//         n(_n), m(_m), nc(_nc), nr(_nr),
//         seed_sz(ss) {

//         insert_kernel = _insert_kernel;
//         search_kernel = _search_kernel;

//         // std::cout << "seed_num: " << seed_num << std::endl;

//         table_total_sz = n * m * WARP_SIZE;
//         // std::cout << "table_total_sz: " << table_total_sz << std::endl;
//         table = gpu_tool<Count_T>.zero(table, table_total_sz);

//         buffer_total_sz = nc * nr;
//         buffer = gpu_tool<Key_T>.zero(buffer, buffer_total_sz);

//         seed_num = 4 * m * WARP_SIZE + 1;
//         seed_total_sz = seed_sz * seed_num;
//         // std::cout << "seed_total_sz: " << seed_total_sz << std::endl;

//         // random seeds
//         seed = gpu_tool<Seed_T>.random(seed, seed_total_sz, 1, std::numeric_limits<Seed_T>::max());
//         // gridDim = dim3(DEFAULT_GRID_DIM_X, 1, 1);
//         // blockDim = dim3(DEFAULT_BLOCK_DIM_X, 1, 1);
//         gridDim = default_grid_dim;
//         blockDim = default_block_dim;

//         nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
//         nwarps = ceil<size_t>(nthreads, WARP_SIZE);

//         // std::cout << "gridDim: " << gridDim << std::endl;
//         // std::cout << "blockDim: " << blockDim << std::endl;

//         // hash = hash_mul_add<Key_T, Hashed_T, Seed_T>;
//         // insert_kernel = insert_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>;

//         cudaStreamCreate(&buffer_steam);
//         cudaStreamCreate(&worker_steam);

//     }

//     virtual int insert(Key_T *keys, size_t keys_sz) {
        
//         return 0;
//     }
//     virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
        
//         return 0;
//     }

//     virtual void clear() {
//         table = gpu_tool<Count_T>.zero(table, table_total_sz);
//     }

//     virtual ~Count_Min_Sketch_GPU() {
//         cudaFree(table);
//         cudaFree(seed);
//     }

// };