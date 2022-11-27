#pragma once

#include <cassert>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <bitset>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include <cuda_runtime.h>
#include <curand.h>



#define ANALYZE_MOD true

#define WARP_SIZE 32ull
#define BUS_WIDTH 11u // 352 bit

// #define DEFAULT_GRID_DIM_X 65536ull
// #define DEFAULT_BLOCK_DIM_X 32ull

#define DEBUG true
#define DEBUG_RANDOM_SEED (DEBUG && false)

extern std::ofstream debug_log;

#define BUFFER_START 1024

#define HASH_MASK_TABLE_SIZE 1024 //
#define HASH_MASK_SIZE 16 // 16 unsigned char

#define HASH_MASK_TABLE_SIZE_SUB_WARP 1024
#define HASH_MASK_SIZE_SUB_WARP 4

#define HASH_MASK_ONES 3
//2097152u


// #define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
//     printf("Error at %s:%d\n",__FILE__,__LINE__);\
//     return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error: %s at %s:%d\n", curandGetStatusString(x),__FILE__,__LINE__);\
    abort();}} while(0)



#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("File %s: Line %d: '%s' failed: %s\n", __FILE__, __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)



template <typename T>
__host__ __device__ T ceil(const T &a, const T &b) {
    return (a + b - 1) / b;
}

__device__ __host__ inline constexpr unsigned int circular_shift_l(unsigned int value, unsigned int count) {
    return value << count | value >> (32 - count);
}
__device__ __host__ inline constexpr unsigned long circular_shift_l(unsigned long value, unsigned long count) {
    return value << count | value >> (64 - count);
}

__device__ __host__ inline constexpr unsigned int circular_shift_r(unsigned int value, unsigned int count) {
    return value >> count | value << (32 - count);
}
__device__ __host__ inline constexpr unsigned long circular_shift_r(unsigned long value, unsigned long count) {
    return value >> count | value << (64 - count);
}


std::ostream &operator<<(std::ostream &os, const dim3 &d);

int read_file(const std::string &file_name, void **buf, size_t &sz);
int write_file(const std::string &file_name, void *buf, size_t sz);

// void *malloc_host(void **ptr, size_t sz);
// void *malloc_device(void *ptr, size_t sz);

// extern std::unordered_set<void *> pointers;
void free_memory();

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
    if (pf < (1ul << 10)) {
        return std::to_string(pf / (1ul)) + " ops/s";
    }
    if (pf < (1ul << 20)) {
        return std::to_string(pf / (1ul << 10)) + " Kops/s";
    }
    return std::to_string(pf / (1ul << 20)) + " Mops/s";
    if (pf < (1ul << 30)) {
        return std::to_string(pf / (1ul << 20)) + " Mops/s";
    }
    
    // return std::to_string(pf / (1ul << 30)) + " Gops/s";
}

template <typename T>
bool is_device_ptr(T *ptr) {
    cudaPointerAttributes att;
    cudaPointerGetAttributes(&att, ptr);
    return att.type == cudaMemoryTypeDevice;
}

// template <typename T>
// bool is_host_ptr(T *ptr) {
//     cudaPointerAttributes att;
//     cudaPointerGetAttributes(&att, ptr);
//     return att.type != cudaMemoryTypeDevice;
// }


template <typename T>
T *copy_to_device(T *ptr, size_t sz) {

    T *tem;
    checkKernelErrors(cudaMalloc((void **)&tem, sizeof(T) * sz));
    checkKernelErrors(cudaMemcpy(tem, ptr, sizeof(T) * sz, cudaMemcpyHostToDevice));
    return tem;
}

template <typename T>
T *copy_to_host(T *ptr, size_t sz) {

    T *tem;
    checkKernelErrors(cudaMallocHost((void **)&tem, sizeof(T) * sz));
    checkKernelErrors(cudaMemcpy(tem, ptr, sizeof(T) * sz, cudaMemcpyDeviceToHost));
    return tem;
}

// template <typename T>
// struct Freq_distribution {
//     // float freq[dis_sz];
//     // T min_val[dis_sz];
//     // T max_val[dis_sz];
//     std::vector<float> freq;
//     // std::vector<T> min_val, max_val;

//     Freq_distribution(const std::vector<float> &f) : freq(f) {
//         float sum = 0.0;
//         for (auto &f : freq) {
//             sum += f;
//         }

//         float cur = 0.0
//         for (auto &f : freq) {
//             cur += f;
//             f = cur / sum;
//         }    
//     }
// };

struct Freq_distribution {
    float low_freq = 0.8;
    size_t low_freq_min = 1;
    size_t low_freq_max = 1;
    size_t high_freq_min = 8;
    size_t high_freq_max = 128;
};

template <typename T>
struct Tools {
    virtual T *random(T *, size_t, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max()) = 0;
    T *read_urandom(T *buf, size_t sz) {
        std::ifstream ifs("/dev/urandom");
        ifs.read((char *)buf, sizeof(T) * sz);
        ifs.close();
        return buf;
    }

    virtual T *random_freq(T *keys, size_t keys_sz, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max(),
        Freq_distribution freq = Freq_distribution()) = 0;

    // virtual T *random_freq(T *keys, size_t keys_sz, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max(),
    //     float low_freq = 0.8, size_t low_freq_min = 1, size_t low_freq_max = 1, size_t high_freq_min =16, size_t high_freq_max = 64) = 0;

    // virtual T *random_freq_count(T *keys, size_t keys_sz, Count_T *counts, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max(),
    //     float low_freq = 0.8, size_t low_freq_min = 1, size_t low_freq_max = 64, size_t high_freq_min =100, size_t high_freq_max = 1000) = 0;

    // virtual T *random_freq(T *keys, size_t keys_sz, const Freq_distribution<T> &, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max()) = 0;

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



template <typename Key_T, typename Count_T> // 
struct Sketch {
    virtual int insert(Key_T *, size_t) = 0;
    virtual int search(Key_T *, size_t, Count_T *) = 0;

    virtual unsigned char *pre_cal(Key_T *, size_t, void *) {
        return nullptr;
    }


    virtual void clear() = 0;
    virtual void print(std::ostream &) {

    }

    virtual ~Sketch() {
        
    }
};





extern const dim3 default_grid_dim;
extern const dim3 default_block_dim;
// #define DEFAULT_GRID_DIM_X 65536ull
// #define DEFAULT_BLOCK_DIM_X 32ull

#include "cpu_hash_functions.h"
#include "host_functions.h"
#include "cpu.h"


#if ANALYZE_MOD

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include "analyze.h"

#endif

#define CONSTANT_SEED_BUFFER_SIZE 1024


#include "gpu_hash_functions.h"
#include "kernel_functions.h"
#include "mem_kernel_functions.h"
#include "mem_kernel_functions_preload.h"
#include "mem_kernel_functions_preload_search.h"
// #include "mem_kernel_functions_preload_buswidth.h"
#include "hash_mask_table_pre_calculation.h"
#include "mem_kernel_functions_pre_calculate_hash_mask_table.h"
#include "mem_kernel_functions_pre_calculate_hash_mask_table_sub_warp.h"
#include "loop.h"
#include "loop2.h"
#include "gpu.h"


#include "test.h"