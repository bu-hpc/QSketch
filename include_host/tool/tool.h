#pragma once

namespace qsketch {

    namespace calculator {

        size_t number_of_buckets(size_t insert_keys_sz, size_t m, double factor = 0.125);
    }

template <typename T>
__host__ __device__ T ceil(const T &a, const T &b) {
    return (a + b - 1) / b;
}

// template <typename T>
// __host__ __device__ size_t bits(size_t sz = 1) {
//     constexpr size_t BITS_PER_BYTE = 8;
//     return (sizeof(T) * BITS_PER_BYTE * sz);
// }
template <typename T>
constexpr size_t bits = sizeof(T) * 8; // BITS_PER_BYTE = 8;

template <typename T>
__device__ const size_t device_bits = sizeof(T) * 8;

template <typename RUN>
double get_run_time(RUN &run) {
    using namespace std::chrono;
    duration<double> time_span;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    
    run();

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    time_span = duration_cast<decltype(time_span)>(t2 - t1);
    return time_span.count();;
}

template <typename RUN>
double get_run_time(const RUN &run) {
    using namespace std::chrono;
    duration<double> time_span;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    
    run();

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    time_span = duration_cast<decltype(time_span)>(t2 - t1);
    return time_span.count();;
}

template <typename T>
std::string format_perf(T pf) {
#if 1
    return std::to_string(pf / (1e6)) + " Mops/s";
    // return std::to_string(pf / (1ul << 20)) + " Mops/s";
#else
    if (pf < (1ul << 10)) {
        return std::to_string(pf / (1ul)) + " ops/s";
    }
    if (pf < (1ul << 20)) {
        return std::to_string(pf / (1ul << 10)) + " Kops/s";
    }
    if (pf < (1ul << 30)) {
        return std::to_string(pf / (1ul << 20)) + " Mops/s";
    }
    return std::to_string(pf / (1ul << 30)) + " Gops/s";
#endif
}

template <typename T>
double format_perf_d(T pf) {
#if 1
    return double(pf / (1ul << 20));
#else
    if (pf < (1ul << 10)) {
        return std::to_string(pf / (1ul)) + " ops/s";
    }
    if (pf < (1ul << 20)) {
        return std::to_string(pf / (1ul << 10)) + " Kops/s";
    }
    if (pf < (1ul << 30)) {
        return std::to_string(pf / (1ul << 20)) + " Mops/s";
    }
    return std::to_string(pf / (1ul << 30)) + " Gops/s";
#endif
}

template <typename T, typename K>
std::string format_bandwidth(T pf) {

    // double time_s = elapsedTimeInMs / 1e3;
    // bandwidthInGBs = (2.0f * memSize * (float)MEMCOPY_ITERATIONS) / (double)1e9;
    // bandwidthInGBs = bandwidthInGBs / time_s;

#if 1
    return std::to_string(2.0f * sizeof(K) * pf / (1e9)) + " GB/s";
#else
    if (pf < (1ul << 10)) {
        return std::to_string(pf / (1ul)) + " ops/s";
    }
    if (pf < (1ul << 20)) {
        return std::to_string(pf / (1ul << 10)) + " Kops/s";
    }
    if (pf < (1ul << 30)) {
        return std::to_string(pf / (1ul << 20)) + " Mops/s";
    }
    return std::to_string(pf / (1ul << 30)) + " Gops/s";
#endif
}

template <typename T>
std::string format_memory_usage(size_t sz) {

    double total_bytes = sizeof(T) * sz;

#if 1
    return std::to_string(total_bytes / (1ul << 20)) + " MB";
#else
    if (total_bytes < (1ul << 10)) {
        return std::to_string(total_bytes / (1ul)) + " B";
    }
    if (total_bytes < (1ul << 20)) {
        return std::to_string(total_bytes / (1ul << 10)) + " KB";
    }
    if (total_bytes < (1ul << 30)) {
        return std::to_string(total_bytes / (1ul << 20)) + " MB";
    }
    return std::to_string(total_bytes / (1ul << 30)) + " GB";
#endif
}






struct Freq_distribution {
    float low_freq = 0.8;
    size_t low_freq_min = 1;
    size_t low_freq_max = 1;
    size_t high_freq_min = 8;
    size_t high_freq_max = 128;
};


template <typename T>
struct Tools {

    // using std::chrono::high_resolution_clock::time_point;

    virtual T *random(T *, size_t, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max()) = 0;
    template <typename U>
    int read_urandom(U *buf, size_t sz) {
        std::ifstream ifs("/dev/urandom");
        if (!ifs) {
            return -1;
        }
        ifs.read((char *)buf, sizeof(U) * sz);
        ifs.close();
        return 0;
    }

    std::chrono::high_resolution_clock::time_point get_current_time() {
        return std::chrono::high_resolution_clock::now();
    }

    virtual T *random_freq(T *keys, size_t keys_sz, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max(),
        Freq_distribution freq = Freq_distribution()) = 0;

    virtual T rand() {
        T r;
        read_urandom(&r, 1);
        return r;
    }
    virtual T *zero(T *, size_t) = 0;

    virtual void random_shuffle(T *, size_t) = 0;

    virtual void free(T *) {

    }
};


// #define seed_buf_sz 100
template <typename T>
struct CPU_Tools : Tools<T> {
    std::default_random_engine eng;
    const static size_t seed_buf_sz = 128;
    char seed_buf[seed_buf_sz];
    CPU_Tools(bool random_seed = true) {
        if (random_seed && RANDOM_SEED) {
            auto start = Tools<T>::get_current_time();
            int r = Tools<T>::read_urandom(seed_buf, seed_buf_sz);
            if (r < 0) {
                std::chrono::high_resolution_clock::duration d = start - Tools<T>::get_current_time();
                eng.seed(d.count());
            } else {
                std::seed_seq sq(seed_buf, seed_buf + seed_buf_sz);
                eng.seed(sq);
            }
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
        // std::cout << "random_freq" << std::endl;
        if (keys == nullptr) {
            keys = new T[keys_sz];
        }

        std::uniform_int_distribution<T> dis(min, max);
        std::uniform_int_distribution<size_t> dis_low(freq.low_freq_min, freq.low_freq_max);
        std::uniform_int_distribution<size_t> dis_high(freq.high_freq_min, freq.high_freq_max);
        std::uniform_real_distribution<float> dis_freq(0.0, 1.0);
        
        size_t i = 0;
        while (i < keys_sz) {
            T rk = dis(eng);
            size_t c;
            if (dis_freq(eng) < freq.low_freq) {
                c = dis_low(eng);
            } else {
                c = dis_high(eng);
            }
            size_t j = 0;
            for (; i < keys_sz && j < c; ++j, ++i) {
                keys[i] = rk;
            }
        }
        
        std::random_shuffle(keys, keys + keys_sz);

        return keys;
    }

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

template <typename T>
struct GPU_Tools : CPU_Tools<T> {

    curandGenerator_t gen;
    GPU_Tools(bool random_seed = true) : CPU_Tools<T>(random_seed) {}

    T *random(T *keys, size_t keys_sz, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max()) {
        T *tem = CPU_Tools<T>::random(nullptr, keys_sz, min, max);
        if (keys == nullptr) {
            CUDA_CALL(cudaMalloc(&keys, sizeof(T) * keys_sz));
        }
        CUDA_CALL(cudaMemcpy(keys, tem, sizeof(T) * keys_sz, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        delete []tem;
        return keys;
    }

    T *random_freq(T *keys, size_t keys_sz, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max(),
        Freq_distribution freq = Freq_distribution()) {
        T *tem = CPU_Tools<T>::random_freq(nullptr, keys_sz, min, max,
            freq);
        if (keys == nullptr) {
            CUDA_CALL(cudaMalloc(&keys, sizeof(T) * keys_sz));
        }
        CUDA_CALL(cudaMemcpy(keys, tem, sizeof(T) * keys_sz, cudaMemcpyHostToDevice));

#if 0 // print the random keys
        {
            for (size_t i = 0; i < keys_sz; ++i) {
                std::cout << "rand: " << tem[i] << std::endl;
            }
        }
#endif

#if 0 // print the count of random keys.
        {
            std::unordered_map<unsigned int, unsigned int> um;
            unsigned int m = 0;
            for (size_t i = 0; i < keys_sz; ++i) {
                // std::cout << tem[i] << std::endl;
                m = std::max(m, ++um[tem[i]]);
            }
            std::cout << "um max: " << m << std::endl;
        }
#endif
        delete []tem;
        return keys;
    }

    T *zero(T *keys, size_t keys_sz) {
        // std::cout << "gpu_tool zero p1" << std::endl;
        if (keys == nullptr) {
            // std::cout << "gpu_tool zero p2" << std::endl;
            // std::cout << "size: " << sizeof(T) * keys_sz << std::endl;
            // here
            CUDA_CALL(cudaMalloc(&keys, sizeof(T) * keys_sz));
            // cudaMalloc(&keys, sizeof(T) * keys_sz);
        }
        // std::cout << "gpu_tool zero p3" << std::endl;
        cudaMemset(keys, 0, sizeof(T) * keys_sz);
        // std::cout << "gpu_tool zero p4" << std::endl;
        return keys;
    }

    T *zero_zero_copy_memory(T **host_table_p, T **table_p, size_t sz) {
        // cudaGetDeviceProperties(&prop, 0);
        //     if (!prop.canMapHostMemory) 
        //         exit(0);
        //     cudaSetDeviceFlags(cudaDeviceMapHost);
        if (host_table_p == nullptr) {
            cudaHostAlloc(host_table_p, sizeof(T) * sz, cudaHostAllocMapped);
            cudaHostGetDevicePointer(table_p, *host_table_p, 0);
        }
        cudaMemset(*table_p, 0, sizeof(T) * sz);
    }


    void random_shuffle(T *keys, size_t keys_sz) {
        // std::random_shuffle(keys, keys + keys_sz);
        // to do 
        // device shuffle kernel
    }

    void free(T *ptr) {
        cudaFree(ptr);
    }
};



/*
gpu rand

curand only supports 32bits int and 64bits int.

*/

template<typename T>
struct Curand_GPU_Tools : GPU_Tools<T> {
    // using T = unsigned int;
    curandGenerator_t gen;
    Curand_GPU_Tools(bool random_seed = true) : GPU_Tools<T>(random_seed) {
        CURAND_CALL(curandCreateGenerator(&gen, 
            CURAND_RNG_PSEUDO_DEFAULT));
        if (random_seed) {
            CURAND_CALL(curandSetGeneratorOffset(gen, this->rand()));
        }
        else {
            CURAND_CALL(curandSetGeneratorOffset(gen, 1234u));
        }
    }

    T *random(T *keys, size_t keys_sz, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max()) {

        if (keys == nullptr) {
            CUDA_CALL(cudaMalloc(&keys, sizeof(unsigned int) * keys_sz));
        }
        CURAND_CALL(curandGenerate(gen, keys, keys_sz));
        cudaDeviceSynchronize();

        if (!(min == std::numeric_limits<T>::min() 
            && max == std::numeric_limits<T>::max())) {
            // to do
        }
        return keys;
    }

    T *random_freq(T *keys, size_t keys_sz, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max(),
        Freq_distribution freq = Freq_distribution()) {
#if 1
        T *tem = CPU_Tools<T>::random_freq(nullptr, keys_sz, min, max,
            freq);

        if (keys == nullptr) {
            CUDA_CALL(cudaMalloc(&keys, sizeof(T) * keys_sz));
        }
        CUDA_CALL(cudaMemcpy(keys, tem, sizeof(T) * keys_sz, cudaMemcpyHostToDevice));

        delete []tem;
        return keys;

#else 
    // to do 
    // use curand to generate freq random number. 

#endif
    }
};

template <typename T>
CPU_Tools<T> cpu_tool;
template <typename T>
GPU_Tools<T> gpu_tool;
template <typename T>
Curand_GPU_Tools<T> curand_gpu_tool;

}

#include "hashmask.h"