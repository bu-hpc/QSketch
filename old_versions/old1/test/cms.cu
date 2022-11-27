#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>

#define HASH hash_mul_add

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

using T = unsigned int;
using C = unsigned int;

using ull = unsigned long long;

const size_t nblocks = 65536;
const size_t threads_per_block = 32;
const size_t nthreads = nblocks * threads_per_block;

const size_t s_sz = 8;
const size_t hts = 65536; // hash_table_size
const size_t nhs = 8;

struct CMS
{
    size_t num_hash_tables = nhs
    size_t hash_table_size = hts;
    size_t seed_size = s_sz;
    T *hash_table;
    T *seed;

    CMS(size_t n1, size_t n2, size_t n3) : num_hash_tables(n1), hash_table_size(n2), seed_size(n3) {
        // hash_table

        checkKernelErrors(cudaMalloc((void **)&hash_table, sizeof(T) * hts * nhs));
        checkKernelErrors(cudaMemset(hash_table, 0, sizeof(T) * hts * nhs));


        checkKernelErrors(cudaMalloc((void **)&seed, sizeof(T) * s_sz * nhs));
    }

    bool set_random_seed() {
        size_t ss = s_sz * nhs;
        T *ts = new T[ss];

        char seed_buf[128];

        std::ifstream ifs("/dev/urandom");

        if (!ifs) {
            return false;
        }

        ifs.read(seed_buf, 128);
        std::seed_seq sq(seed_buf, seed_buf + 128);

        std::default_random_engine gen(sq);
        std::uniform_int_distribution<T> dis(1, std::numeric_limits<T>::max());

        for (int i = 0; i < ss; ++i)
        {
            ts[i] = (dis(gen));
            std::cout << ts[i] << ",";
        }
        std::cout << std::endl;

        checkKernelErrors(cudaMemcpy(seed, ts, ss * sizeof(T), cudaMemcpyHostToDevice));

        delete [] ts;
        ifs.close();

        return true;
    }

    T *copy_hash_table_to_host() {
        size_t hs = hts * nhs;
        T *hht = new T[hs];
        checkKernelErrors(cudaMemcpy(hht, hash_table, hs * sizeof(T), cudaMemcpyDeviceToHost));

        return hht;
    }
};


__device__ T hash_mul(T *seed, T s_sz, const T &key) {
    T hv = key;
    for (size_t i = 0; i < s_sz; ++i)
    {
        hv = hv * seed[i];
    }
    return hv;
}

__device__ T hash_mul_add(T *seed, T s_sz, const T &key) {
    T hv = 0;
    for (size_t i = 0; i < s_sz; ++i)
    {
        hv += key * seed[i];
    }
    return hv;
}


__global__ void insert(CMS *cms, T *val, T sz, T ptm, void *debug) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = tid * ptm;
    size_t e = (tid + 1) * ptm;
    

    for (size_t i = b; i < e; ++i)
    {
        for (size_t j = 0; j < cms->nhs; ++j) {
            T hv = HASH(cms->seed + j * cms->seed_size, cms->seed_size, val[i]) % cms->hts;
            T *address = cms->hash_table + j * cms->hts + hv;
            atomicAdd(address, 1);
        }
        
    }
}




int main(int argc, char const *argv[])
{
    size_t tm = std::numeric_limits<T>::max();
    size_t sz = 65536;
    size_t ptm = (tm + 1) / nthreads;

    // std::cout << "tm: \t" << tm << std::endl;
    // std::cout << "ptm: \t" << ptm << std::endl;


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