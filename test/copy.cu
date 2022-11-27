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


#define CUDA_CALL(expr)                                             \
    do {                                                            \
        expr;                                                       \
        cudaError_t __err = cudaGetLastError();                     \
        if((__err) != cudaSuccess) {                                \
            printf("Error at %s:%d:'%s','%s'\n",__FILE__, __LINE__, \
                #expr, cudaGetErrorString(__err));                  \
            abort();                                                \
        }                                                           \
    } while(0)

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

template <typename Count_T>
struct Device_Hash_Table {
    size_t n = 0; // the number of hash tables
    size_t m = 0; // the size of each hash table
    size_t table_total_sz = 0;
    Count_T *table = nullptr;
    uint *next_level_id = nullptr;

    Device_Hash_Table() = default;
    Device_Hash_Table(size_t _n, size_t _m) {
        resize(_n, _m);
    }

    void resize(size_t _n, size_t _m) {
        n = _n;
        m = _m;
    }
    void clear() {
        
    }

    ~Device_Hash_Table() {
        cudaFree(table);
        cudaFree(next_level_id);
    }
};

template <typename T>
__global__ void test(size_t keys_sz, Device_Hash_Table<T> dht) {
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    if (tid == 0) {
        printf("dht: %lu, %lu\n", dht.n, dht.m);
    }
}

template <typename T>
struct Run_Base
{
    virtual void main() {
        
    }
};

template <typename T>
struct Run : Run_Base<T>
{
    Device_Hash_Table<unsigned int> dht;
    void main() {
        for (int i = 0; i < 32; ++i) {
            CUDA_CALL((test<unsigned int><<<1, 32>>>(100, dht)));
            cudaDeviceSynchronize();
        }
    }
};

int main(int argc, char const *argv[])
{
    
    
    Run<unsigned int> r;
    r.dht.resize(1, 3);

    Run_Base<unsigned int> &rb = r;
    rb.main();

    return 0;
}

