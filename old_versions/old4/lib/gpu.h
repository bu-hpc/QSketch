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
};

template <>
GPU_Tools<unsigned int>::GPU_Tools(bool random_seed) : CPU_Tools<unsigned int>(random_seed) {
    CURAND_CALL(curandCreateGenerator(&gen, 
            CURAND_RNG_QUASI_SOBOL32));
    CURAND_CALL(curandSetGeneratorOffset(gen, this->rand()));
}

template <>
unsigned int * GPU_Tools<unsigned int>::random(unsigned int *keys, size_t keys_sz, 
    unsigned int min, unsigned int max) {

    // std::cout << "gpu random, unsigned int, " << keys_sz << std::endl;
    if (keys == nullptr) {
        checkKernelErrors(cudaMalloc(&keys, sizeof(unsigned int) * keys_sz));
    }
    CURAND_CALL(curandGenerate(gen, keys, keys_sz));
    cudaDeviceSynchronize();
    return keys;
}

template <>
GPU_Tools<unsigned long long>::GPU_Tools(bool random_seed) : CPU_Tools<unsigned long long>(random_seed) {
    CURAND_CALL(curandCreateGenerator(&gen, 
            CURAND_RNG_QUASI_SOBOL64));
    CURAND_CALL(curandSetGeneratorOffset(gen, this->rand()));
}

template <>
unsigned long long * GPU_Tools<unsigned long long>::random(unsigned long long *keys, size_t keys_sz, 
    unsigned long long min, unsigned long long max) {

    if (keys == nullptr) {
        checkKernelErrors(cudaMalloc(&keys, sizeof(unsigned long long) * keys_sz));
    }
    CURAND_CALL(curandGenerateLongLong(gen, keys, keys_sz));
    cudaDeviceSynchronize();
    return keys;
}

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

template <typename Key_T, typename Count_T, typename Hashed_T = size_t, typename Seed_T = size_t> // 
struct Count_Min_Sketch_GPU : Sketch<Key_T, Count_T> {

    dim3 gridDim;
    dim3 blockDim;

    size_t nwarps;
    size_t nthreads;

    size_t n;
    size_t m;
    size_t table_total_sz;
    Count_T *table = nullptr;

    // Device_Data *device_hash_table;

    size_t seed_sz;
    size_t seed_num;
    size_t seed_total_sz;

    // Device_Data *device_seed;
    Seed_T *seed = nullptr;
    // Hash<Seed_T> *hs = nullptr;

    Count_Min_Sketch_GPU(size_t _n = default_hash_n, size_t _m = default_hash_m,
        size_t ss = default_seed_sz) : 
        n(_n), m(_m),
        seed_sz(ss) {

        seed_num = m * WARP_SIZE + 1;
        // std::cout << "seed_num: " << seed_num << std::endl;

        table_total_sz = n * m * WARP_SIZE;
        // std::cout << "table_total_sz: " << table_total_sz << std::endl;
        table = gpu_tool<Count_T>.zero(table, table_total_sz);

        seed_total_sz = seed_sz * seed_num;
        // std::cout << "seed_total_sz: " << seed_total_sz << std::endl;

        // random seeds
        seed = gpu_tool<Seed_T>.random(seed, seed_total_sz, 1, std::numeric_limits<Seed_T>::max());
        gridDim = dim3(DEFAULT_GRID_DIM_X, 1, 1);
        blockDim = dim3(DEFAULT_BLOCK_DIM_X, 1, 1);

        nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
        nwarps = ceil<size_t>(nthreads, WARP_SIZE);

        std::cout << "gridDim.x: " << gridDim.x << std::endl;
        std::cout << "blockDim.x: " << blockDim.x << std::endl;
    }

    virtual int insert(Key_T *keys, size_t keys_sz) {
        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        checkKernelErrors((insert_warp<<<gridDim, blockDim>>>(
                keys, keys_sz,
                table, n, m,
                COUNT_MIN_SKETCH_GPU_HASH_FUNCTION<Key_T, Hashed_T, Seed_T>(),
                seed, seed_sz,
                work_load_per_warp,
                nullptr
            )));
        cudaDeviceSynchronize();
        // std::cout << "GPU insert" << std::endl;
        return 0;
    }
    virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
        // for (size_t i = 0; i < keys_sz; ++i) {
        //     count[i] = um[keys[i]];
        // }
        return 0;
    }

    virtual void clear() {

    }
};
