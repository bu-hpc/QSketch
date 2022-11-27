/*
 * This program uses the host CURAND API to generate 100 
 * pseudorandom floats.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <thrust/sort.h>
#include <iostream>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <chrono>

#define WARP_SIZE 32

using T = unsigned int; // the type of key and mapped
using C = unsigned int; // the type of count

using ull = unsigned long long;

const size_t s_sz = 2;
const size_t hash_table_sz = 65521;
const size_t hash_table_num = 16;

// const size_t nblocks = 65536;
const size_t nblocks = 1;
const size_t threads_per_block = 32;
const size_t nthreads = nblocks * threads_per_block;
const size_t nwarps = (nthreads + WARP_SIZE - 1) / WARP_SIZE;

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error: %s at %s:%d\n", curandGetStatusString(x),__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)


#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)


// using ulong = unsigned long long;

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

__global__ void tran(unsigned long long *src, size_t sz, unsigned long long *dec, size_t work_load_per_warp) {
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;
    // e = (e < sz) ? (e) : sz;

    for (size_t i = b; i < e; i += WARP_SIZE) {
        size_t t = i + tid;
        if (t < sz) {
            dec[src[t] % sz] = src[t];
            // printf("%lu\n", t);
        }

        // for (int j = 0; j < hash_table_num; ++j) {
        //     T hash_val = hash2(seed + j * s_sz, s_sz, keys[t]) % hash_table_sz;
        //     atomicAdd(hash_table + j * hash_table_sz + hash_val, 1);
        // }
    }
}

int main(int argc, char *argv[])
{
    size_t n = 128 * 1024 *1024;
    // size_t i;
    size_t l = 1;
    curandGenerator_t gen;
    unsigned long long *devData, *hostData;
    unsigned long long *bufData;

    /* Allocate n floats on host */
    // hostData = (float *)calloc(n, sizeof(float));
    hostData = new unsigned long long[n];

    /* Allocate n floats on device */
    CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(unsigned long long)));
    CUDA_CALL(cudaMalloc((void **)&bufData, n*sizeof(unsigned long long)));

    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_QUASI_SOBOL64));
    // CURAND_CALL(curandCreateGenerator(&gen, 
    //             CURAND_RNG_PSEUDO_XORWOW));
    
    /* Set seed */
    // CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
    //             1234ULL));

    // offset
    CURAND_CALL(curandSetGeneratorOffset(gen, 1234ULL));

    /* Generate n floats on device */
    CURAND_CALL(curandGenerateLongLong(gen, devData, n));

    {
        using namespace std::chrono;
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        for (size_t i = 0; i < l; ++i)
        {
            CURAND_CALL(curandGenerateLongLong(gen, devData, n));
            thrust::sort(thrust::device, devData, devData + n);
        }
        cudaDeviceSynchronize();
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        size_t total_bytes = sizeof(unsigned long long) * n * l;
        // std::cout << "total_bytes: " << total_bytes << "-->" << (total_bytes / (1ul<<30)) << std::endl;
        std::cout << "perf: " << double(total_bytes) / time_span.count() / (1ul<<30) << "GB/s" << std::endl;

    }

    {
        size_t work_load_per_warp = (n + (nwarps - 1)) / nwarps;
        std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;
        using namespace std::chrono;
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        for (size_t i = 0; i < l; ++i)
        {
            CURAND_CALL(curandGenerateLongLong(gen, devData, n));
            checkKernelErrors((tran<<<nblocks, threads_per_block>>>(devData, n, bufData, work_load_per_warp)));
        }
        cudaDeviceSynchronize();
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        size_t total_bytes = sizeof(unsigned long long) * n * l;
        // std::cout << "total_bytes: " << total_bytes << "-->" << (total_bytes / (1ul<<30)) << std::endl;
        std::cout << "perf: " << double(total_bytes) / time_span.count() / (1ul<<30) << "GB/s" << std::endl;

    }
    // thrust::sort(thrust::device, devData, devData + n);

    




    /* Copy device memory to host */
    // CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(unsigned long long),
    //     cudaMemcpyDeviceToHost));

    // /* Show result */
    // for(i = 0; i < n; i++) {
    //     printf("%llu ", hostData[i]);
    // }
    // printf("\n");

    // for (int i = 1; i < n; ++i)
    // {
    //     if (hostData[i - 1] > hostData[i]) {
    //         printf("unsorted\n");
    //     }
    // }

    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
    delete [] hostData;
    return EXIT_SUCCESS;
}