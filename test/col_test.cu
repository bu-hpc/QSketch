#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)

using T = unsigned int;
using C = unsigned int;

using ull = unsigned long long;

const size_t s_sz = 2;

const size_t nblocks = 65536;
const size_t threads_per_block = 32;
const size_t nthreads = nblocks * threads_per_block;

__device__ T hash(T *seed, T s_sz, const T &key) {
    T hv = key;
    for (size_t i = 0; i < s_sz; ++i)
    {
        hv = hv * seed[i];
    }
    return hv;
}


__global__ void col(T *count, T sz, T *seed, T s_sz, T ptm, ull *c) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t e = (tid + 1) * ptm;
    atomicAdd(c, ull(1));
    for (size_t i = tid * ptm; i < e; ++i)
    {
        // atomicAdd(c, ull(1));
        atomicAdd(&count[hash(seed, s_sz, i) % sz], 1);
        // ++count[i % sz];
    }

    // if (threadIdx.x == 0) {
    //     printf("%u:%u\n", blockIdx.x, count[blockIdx.x]);
    // }
}


int main(int argc, char const *argv[])
{
    size_t tm = std::numeric_limits<T>::max();
    size_t sz = 65536;
    size_t ptm = (tm + 1) / nthreads;

    std::cout << "tm: \t" << tm << std::endl;
    std::cout << "ptm: \t" << ptm << std::endl;
    T *seed;
    checkKernelErrors(cudaMalloc((void **)&seed, sizeof(T) * s_sz));
    {
        
        
        T *ts = new T[s_sz];
        std::default_random_engine gen;
        std::uniform_int_distribution<T> dis(1, std::numeric_limits<T>::max());

        for (int i = 0; i < s_sz; ++i)
        {
            ts[i] = (dis(gen));
            std::cout << ts[i] << ",";
        }
        std::cout << std::endl;

        checkKernelErrors(cudaMemcpy(seed, ts, s_sz * sizeof(T), cudaMemcpyHostToDevice));

        delete [] ts;
    }

    ull *c;
    checkKernelErrors(cudaMalloc((void **)&c, sizeof(ull)));
    checkKernelErrors(cudaMemset(c, 0, sizeof(ull)));


    T *count = (T *)malloc(sizeof(T) * sz);
    if (count == nullptr) {
        std::cout << "error" << std::endl;
    }
    T *d_count;
    checkKernelErrors(cudaMalloc((void **)&d_count, sizeof(T) * sz));
    checkKernelErrors(cudaMemset(d_count, 0, sizeof(T) * sz));

    checkKernelErrors((col<<<nblocks, threads_per_block>>>(d_count, sz, seed, s_sz, ptm, c)));
    // checkKernelErrors((test<<<32, 32>>>(d_count)));


    checkKernelErrors(cudaMemcpy(count, d_count, sizeof(T) * sz, cudaMemcpyDeviceToHost));


    {
        size_t total = 0;
        T max = 0;
        for (size_t i = 0; i < sz; ++i) {
            total += count[i];
            max = std::max(max, count[i]);
        }
        std::cout << "total: \t" << total << std::endl;
        std::cout << "max: \t" << max << std::endl;

        ull hc;
        checkKernelErrors(cudaMemcpy(&hc, c, sizeof(ull), cudaMemcpyDeviceToHost));
        std::cout << "hc: \t" << hc << std::endl;

        // std::cout << "ave: \t" << 
    }

    delete [] count;
    cudaFree(seed);
    cudaFree(d_count);

    return 0;
}