#include <cuda_runtime.h>
#include <cuda.h>
// #include <crt/sm_80_rt.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <chrono>

#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)

// to do
// use 8bit unsigned char

#define WARP_SIZE 32ul

using T = unsigned int; // the type of key and mapped
using C = unsigned int; // the type of count
C C_MAX = std::numeric_limits<C>::max();

using ull = unsigned long long;

const size_t nblocks = 65536;
// const size_t nblocks = 1;
const size_t threads_per_block = 32;
const size_t nthreads = nblocks * threads_per_block;
const size_t nwarps = (nthreads + WARP_SIZE - 1) / WARP_SIZE;


// const size_t hash_table_sz = 65521;
// const size_t hash_table_num = 16;
const size_t n = 4093;
const size_t n_alignment = 4096;
const size_t m = 8;
// const size_t k = 32;

const size_t nseed = m * WARP_SIZE + 1;
const size_t s_sz = 2;

__global__ void test_v1(T *keys, size_t sz,  size_t work_load_per_warp) 
{

    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    __shared__ T v[WARP_SIZE];
    __shared__ T h[WARP_SIZE];

    for (size_t i = b; i < e; i += WARP_SIZE) {
        size_t t = i + tid;
        v[tid] = keys[t];
        h[tid] = keys[t];
        v[tid]++;
        // h[tid]++;

        
    }
}


__global__ void test_v2(T *keys, size_t sz,  size_t work_load_per_warp) 
{

    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    __shared__ T v[WARP_SIZE];
    __shared__ T h[WARP_SIZE];

    for (size_t i = b; i < e; i += WARP_SIZE) {
        size_t t = i + tid;
        v[tid] = keys[t];
        v[tid]++;
        h[tid] = keys[t];
        // h[tid]++;
    }
}


int main(int argc, char const *argv[])
{
    size_t keys_sz = 1024 * 1024 * 1024;
    size_t l = 10;
    T *keys;
    checkKernelErrors(cudaMalloc((void **)&keys, sizeof(T) * keys_sz));
    cudaDeviceSynchronize();

    {

        size_t work_load_per_warp = (keys_sz + (nwarps - 1)) / nwarps;
        std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;
        using namespace std::chrono;
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        
        for (size_t i = 0; i < l; ++i)
        {
            checkKernelErrors((test_v1<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp)));   
        }
        cudaDeviceSynchronize();

        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        // size_t total_bytes = sizeof(T) * keys_sz * hash_table_num * insert_loop;
        size_t insert_number = keys_sz * l;
        // std::cout << "total_bytes: " << total_bytes << "-->" << (total_bytes / (1ul<<30)) << std::endl;
        // std::cout << "test_v1 perf: " << double(insert_number) / time_span.count() / (1ul<<20) << "Mops/s" << std::endl;
        std::cout << "test_v1 perf: " << double(insert_number) / time_span.count() / (1ul<<30) << "Gops/s" << std::endl;

    }

    {

        size_t work_load_per_warp = (keys_sz + (nwarps - 1)) / nwarps;
        std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;
        using namespace std::chrono;
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        
        for (size_t i = 0; i < l; ++i)
        {
            checkKernelErrors((test_v2<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp)));   
        }
        cudaDeviceSynchronize();

        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        // size_t total_bytes = sizeof(T) * keys_sz * hash_table_num * insert_loop;
        size_t insert_number = keys_sz * l;
        // std::cout << "total_bytes: " << total_bytes << "-->" << (total_bytes / (1ul<<30)) << std::endl;
        std::cout << "test_v2 perf: " << double(insert_number) / time_span.count() / (1ul<<30) << "Gops/s" << std::endl;

    }

    return 0;
}