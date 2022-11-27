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
    using T = unsigned int;

__global__ void test(T *table, size_t sz) {
    if (threadIdx.x == 0) {
        for (size_t i = 0; i < sz; ++i) {
            // table[i] = i;
            atomicAdd(table + i, 1);
            printf("%u\n", table[i]);
        }
    }
}

int main(int argc, char const *argv[])
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("cuda flag: %d\n", int(prop.canMapHostMemory));
        if (!prop.canMapHostMemory) 
            exit(0);
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaDeviceSynchronize();

    size_t sz = 1024;
    T *host_table = nullptr;
    T *table = nullptr;

    size_t mem_sz = 1610610944;

    checkKernelErrors(cudaMallocManaged(&table, mem_sz));
    // checkKernelErrors(cudaHostAlloc(&host_table, mem_sz, cudaHostAllocMapped));
    // checkKernelErrors(cudaHostGetDevicePointer(&table, host_table, 0));
    // checkKernelErrors(cudaMemset(table, 0, mem_sz));
    test<<<1, 32>>>(table, sz);
    cudaDeviceSynchronize();
}
