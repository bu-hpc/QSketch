#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
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
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#define WORKLOAD_PERTHREAD 102400
#define CONSTANT_SIZE 10240

#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("File %s: Line %d: '%s' failed: %s\n", __FILE__, __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error: %s at %s:%d\n", curandGetStatusString(x),__FILE__,__LINE__);\
    abort();}} while(0)




const char* curandGetStatusString(curandStatus_t status) {
// detail info come from http://docs.nvidia.com/cuda/curand/group__HOST.html
    switch(status) {
        case CURAND_STATUS_SUCCESS:                     return "CURAND_STATUS_SUCCESS";
        case CURAND_STATUS_VERSION_MISMATCH:            return "CURAND_STATUS_VERSION_MISMATCH";
        case CURAND_STATUS_NOT_INITIALIZED:             return "CURAND_STATUS_NOT_INITIALIZED";
        case CURAND_STATUS_ALLOCATION_FAILED:           return "CURAND_STATUS_ALLOCATION_FAILED";
        case CURAND_STATUS_TYPE_ERROR:                  return "CURAND_STATUS_TYPE_ERROR";
        case CURAND_STATUS_OUT_OF_RANGE:                return "CURAND_STATUS_OUT_OF_RANGE";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:         return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:   return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        case CURAND_STATUS_LAUNCH_FAILURE:              return "CURAND_STATUS_LAUNCH_FAILURE";
        case CURAND_STATUS_PREEXISTING_FAILURE:         return "CURAND_STATUS_PREEXISTING_FAILURE";
        case CURAND_STATUS_INITIALIZATION_FAILED:       return "CURAND_STATUS_INITIALIZATION_FAILED";
        case CURAND_STATUS_ARCH_MISMATCH:               return "CURAND_STATUS_ARCH_MISMATCH";
        case CURAND_STATUS_INTERNAL_ERROR:              return "CURAND_STATUS_INTERNAL_ERROR";
    }
    return "CURAND_STATUS_UNKNOWN_ERROR";
}


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
    
    return std::to_string(pf / (1ul << 30)) + " Gops/s";
}


template <typename T>
T ceil(const T &a, const T &b) {
    return (a + b - 1) / b;
}

#define WARP_SIZE 32
#define BUS_WIDTH 11

template<typename T>
__global__ void test(unsigned int *keys, size_t sz, size_t work_load_per_warp,
    unsigned int *buf, size_t buf_sz, size_t tm, size_t rm) {
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;
    if (e >= sz) {
        e = sz;
    }

    size_t lm = (buf_sz - rm) / BUS_WIDTH;

    for (size_t i = b; i < e; i++) {
        unsigned int kv = keys[i];
        unsigned int *ptr = buf + (kv % lm) * BUS_WIDTH + rm;
        // ptr = buf + (kv % sz);

        // if ((kv + tid) % 8 == 0) {
        //     // atomicAdd(ptr + tid % tm, 1);
        //     ptr[tid%tm] = kv;
        // }
        // if (tid < 2) {
        //     ptr[tid * tm + (kv % tm)] = kv;
        // }

        if (tid < tm) 
        // if (kv % 1000007 == 0)
        {
            // atomicAdd(ptr + tid, 1);
            if (kv + tid)
            ptr[tid] = kv;
        }
    }
}
__device__ unsigned long long ans;

int main(int argc, char const *argv[])
{
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, 
            CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetGeneratorOffset(gen, 1024));


    unsigned int *keys;
    unsigned int *buf;
    size_t keys_sz = 1024 * 1024 * 1024;
    size_t buf_sz = 128 * 1024 * 1024;
    checkKernelErrors(cudaMalloc(&keys, sizeof(unsigned int) * keys_sz));
    checkKernelErrors(cudaMalloc(&buf, sizeof(unsigned int) * buf_sz));
    CURAND_CALL(curandGenerate(gen, keys, keys_sz));
    
    int nblocks = 65536;
    int threads_per_block = 32;
    int nthreads = nblocks * threads_per_block;
    // double workload = double(WORKLOAD_PERTHREAD) * nthreads;
    double workload = keys_sz;
    size_t work_load_per_warp = ceil<size_t>(keys_sz, nblocks);
    // for (unsigned int i = 0; i < BUS_WIDTH; ++i) {
    //     double ts = get_run_time([&](){
    //         test<unsigned int><<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp, buf, buf_sz, 32, i);
    //         cudaDeviceSynchronize();
    //     });
    //     // std::cout << ts << std::endl;
    //     // std::cout << i << " perf: " << format_perf(workload / ts) << std::endl;
    
    // }
    test<unsigned int><<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp, buf, buf_sz,2, 2);
                cudaDeviceSynchronize();

    for (unsigned int j = 1; j <= 16; ++j) {
        size_t work_load_per_warp = ceil<size_t>(keys_sz, nblocks);
        double workload = keys_sz * 4;
        // for (unsigned int i = 0; i < BUS_WIDTH; ++i) 
        unsigned int i = 0;
        {
            double ts = get_run_time([&](){
                test<unsigned int><<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp, buf, buf_sz,j, i);
                cudaDeviceSynchronize();
            });
            std::cout << ts << std::endl;
            std::cout << "tm: " << j << "\trm: " << i;
            std::cout << "\tperf: " << format_perf(workload / ts) << std::endl;
        
        }
    }
    
    

    // std::cout << WORKLOAD_PERTHREAD << std::endl;
    // std::cout << nthreads << std::endl;
    

    return 0;
}