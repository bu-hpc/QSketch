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

        // if (keys_sz == 2056) {
        //     for (size_t i = 0; i < keys_sz; ++i) {
        //         std::cout << "seed: " << tem[i] << std::endl;
        //     }
        // }
        delete []tem;
        return keys;
    }

    T *zero(T *keys, size_t keys_sz) {

        // std::cout << "cpu zero" << std::endl;

        if (keys == nullptr) {
            checkKernelErrors(cudaMalloc(&keys, sizeof(T) * keys_sz));
        }
        cudaMemset(keys, 0, sizeof(T) * keys_sz);
        return keys;
    }

    void random_shuffle(T *keys, size_t keys_sz) {
        // std::random_shuffle(keys, keys + keys_sz);
    }
};

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

constexpr size_t host_buf_sz = 1024 * 1024;


template <typename Key_T, typename Count_T, typename Hashed_T = size_t> // 
struct Unordered_Map_GPU : Unordered_Map<Key_T, Count_T, Hashed_T> {
    // using Unordered_Map<Key_T, Count_T, Hashed_T>::um;
    virtual int insert(Key_T *keys, size_t keys_sz) {

        static Key_T buf[host_buf_sz];

        for (size_t i = 0; i < keys_sz; i += host_buf_sz) {
            size_t cpd = std::min(host_buf_sz, keys_sz - i);
            checkKernelErrors(cudaMemcpy(buf, keys + i, sizeof(Key_T) * cpd, cudaMemcpyDeviceToHost));
            // ++um[keys[i]];
            Unordered_Map<Key_T, Count_T, Hashed_T>::insert(buf, cpd);
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

#ifndef COUNT_MIN_SKETCH_GPU_HASH_FUNCTION
#define COUNT_MIN_SKETCH_GPU_HASH_FUNCTION hash_mul_add
#endif

// template <typename T>
// struct Device_Data {
//     // n small tables, each table is T[m]
//     size_t m;
//     size_t n;
//     size_t sz; // sz == m * n
//     T *data;

//     Device_Data() = default;

//     Device_Data(size_t _m, size_t _n, size_t _sz) : m(_m), n(_n), sz(_sz) {
//         checkKernelErrors(cudaMalloc(&data, sizeof(T) * sz));
//     }
// };

template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, 
    typename Hash_Function> // 
struct Count_Min_Sketch_GPU : Sketch<Key_T, Count_T> {

    // using Hash_Function = __device__ __host__ Hashed_T (*)(Seed_T *, size_t, const Key_T &);
    // using Hash_Function = Hashed_T (Seed_T *, size_t, const Key_T &);


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

        seed_num = m * WARP_SIZE + 1;
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

};
