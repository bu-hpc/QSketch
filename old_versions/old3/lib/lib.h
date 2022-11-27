#pragma once

#include <cassert>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <cuda_runtime.h>


#define ANALYZE_MOD true

#define WARP_SIZE 32ul
#define DEFUALT_GRID_DIM_X 65536
#define DEFAULT_BLOCK_DIM_X 32

#define DEBUG true
std::ofstream debug_log("log.txt");



#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("File %s: Line %d: '%s' failed: %s\n", __FILE__, __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)


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
    if (pf < (1ul << 30)) {
        return std::to_string(pf / (1ul << 20)) + " Mops/s";
    }
    
    return std::to_string(pf / (1ul << 30)) + " Gps/s";
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

template <typename T>
struct Tools {
    virtual T *random(T *, size_t, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max()) = 0;
    virtual T *zero(T *, size_t) = 0;

};



template <typename Key_T, typename Count_T> // 
struct Sketch {
    virtual int insert(Key_T *, size_t) = 0;
    virtual int search(Key_T *, size_t, Count_T *) = 0;
    virtual void clear() = 0;
    virtual void print(std::ostream &) {

    }
};


#include "cpu.h"
#include "gpu.h"

// #if ANALYZE_MOD

#include <thrust/device_vector.h>
// #include <thrust/transform.h>
// #include <thrust/functional.h>
// #include <thrust/reduce.h>
// #include "analyze.h"

template <typename C>
thrust::device_vector<C> diff(const thrust::device_vector<C> &dc1, const thrust::device_vector<C> &dc2) {
    // dc1.size();
    // thrust::device_vector<int> dc3(100ul);

    // thrust::device_vector<int> dc3(dc1.begin(), dc1.end());
    // thrust::device_vector<C> dc3(dc1.size());
    // thrust::transform(dc1.begin(), dc1.end(), dc2.begin(), dc3.begin(), thrust::minus<C>());

    thrust::device_vector<C> ans;
    return ans;
    // return dc3;
}

// #endif

#include "test.h"