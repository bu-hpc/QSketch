#include <cuda_runtime.h>

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

using ull = unsigned long long;

const size_t s_sz = 2;
const size_t hash_table_sz = 65521;
const size_t hash_table_num = 16;

const size_t nblocks = 65536;
// const size_t nblocks = 1;
const size_t threads_per_block = 32;
const size_t nthreads = nblocks * threads_per_block;
const size_t nwarps = (nthreads + WARP_SIZE - 1) / WARP_SIZE;

__device__ T hash(T *seed, T s_sz, const T &key) {
    T hv = key;
    for (size_t i = 0; i < s_sz; ++i)
    {
        hv = hv * seed[i];
    }
    return hv;
}

__device__ T hash2(T *seed, T s_sz, const T &key) {
    T hv = 0;
    for (size_t i = 0; i < s_sz; ++i)
    {
        hv += key * seed[i];
    }
    return hv;
}

__device__ T hash3(T *seed, T s_sz, const T &key) {
    T hv = 0;
    T k = key;
    for (size_t i = 0; i < s_sz; ++i)
    {
        hv += k * seed[i];
        k *= key;
    }
    return hv;
}

__device__ T hash4(T *seed, T *seed2, T s_sz, const T &key) {
    T hv = 0;
    for (size_t i = 0; i < s_sz; ++i)
    {
        hv += (key + seed2[i]) * seed[i];
    }
    return hv;
}

// __device__ T hash_cublas(T *seed, T s_sz, T *key) {

// }

// __device__ T 

__global__ void insert(T *keys, size_t sz,  size_t work_load_per_thread,
    C *hash_table, size_t hash_table_num,  size_t hash_table_sz,
    T *seed,                               size_t s_sz, 
    void *debug)
{

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = tid * work_load_per_thread;
    size_t e = (tid + 1) * work_load_per_thread;
    e = (e <= sz) ? e : sz;

    // printf("b: %lu, e: %lu\n", b, e);

    for (size_t i = b; i < e; ++i) {
        // printf("i: %lu\n", i);
        for (int j = 0; j < hash_table_num; ++j) {
            T hash_val = hash2(seed + j * s_sz, s_sz, keys[i]) % hash_table_sz;
            atomicAdd(hash_table + j * hash_table_sz + hash_val, 1);
        }
    }
}


__global__ void insert2(T *keys, size_t sz,  size_t work_load_per_warp,
    C *hash_table, size_t hash_table_num,  size_t hash_table_sz,
    T *seed,                               size_t s_sz, 
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    for (size_t i = b; i < e; i += WARP_SIZE) {
        size_t t = i + tid;
        // printf("%lu\n", t);
        for (int j = 0; j < hash_table_num; ++j) {
            T hash_val = hash2(seed + j * s_sz, s_sz, keys[t]) % hash_table_sz;
            atomicAdd(hash_table + j * hash_table_sz + hash_val, 1);
        }
    }
}

__global__ void insert3(T *keys, size_t sz,  size_t work_load_per_warp,
    C *hash_table, size_t hash_table_num,  size_t hash_table_sz,
    T *seed,                               size_t s_sz, 
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    for (size_t i = b; i < e; i += WARP_SIZE) {
        size_t t = i + tid;
        // printf("%lu\n", t);
        T hid = hash2(seed, s_sz, keys[t]) % hash_table_num;
        for (int j = 0; j < hash_table_sz; ++j) {
            // T hash_val = hash2(seed + j * s_sz, s_sz, keys[t]) % hash_table_sz;
            // atomicAdd(hash_table + hid * hash_table_sz + hash_val, 1);
            // atomicAdd(hash_table + hid * hash_table_sz + j, 1);
        }
    }
}


__global__ void insert4(T *keys, size_t sz,  size_t work_load_per_warp,
    C *hash_table, size_t hash_table_num,  size_t hash_table_sz,
    T *seed,                               size_t s_sz, 
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    for (size_t i = b; i < e; i += WARP_SIZE) {
        size_t t = i + tid;
        // printf("%lu\n", t);
        __shared__ T hid;
        T v = keys[t];
        if (tid == 0) {
            hid = hash2(seed, s_sz, v) % hash_table_num;
        }

        for (size_t j = 0; j < hash_table_sz; j += WARP_SIZE) {
            // if (hash2(seed + tid * s_sz, s_sz, v) % (1) == 1) {
                hash2(seed + tid * s_sz, s_sz, v) % (1);
                atomicAdd(hash_table + hid * hash_table_sz + j, 1);
            // }
        }
        

        // T val =
        // for (int j = 0; j < hash_table_sz; ++j) {
        //     // T hash_val = hash2(seed + j * s_sz, s_sz, keys[t]) % hash_table_sz;
        //     // atomicAdd(hash_table + hid * hash_table_sz + hash_val, 1);
        //     // atomicAdd(hash_table + hid * hash_table_sz + j, 1);
        // }
    }
}



__global__ void search(T *keys, size_t sz,  size_t work_load_per_thread,
    C *hash_table, size_t hash_table_num,  size_t hash_table_sz,
    T *seed,                               size_t s_sz,
    C *count,
    void *debug)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = tid * work_load_per_thread;
    size_t e = (tid + 1) * work_load_per_thread;
    e = (e <= sz) ? e : sz;

    for (size_t i = b; i < e; ++i) {
        // printf("i\n");
        int j = 0;
        T hash_val = hash2(seed + j * s_sz, s_sz, keys[i]) % hash_table_sz;
        C min_c = hash_table[j * hash_table_sz + hash_val];
        ++j;
        for (; j < hash_table_num; ++j) {
            T hash_val = hash2(seed + j * s_sz, s_sz, keys[i]) % hash_table_sz;
            C tem = hash_table[j * hash_table_sz + hash_val];
            min_c = (tem < min_c) ? tem : min_c;
        }
        count[i] = min_c;
    }
}

__global__ void random(T *keys, size_t sz, size_t work_load_per_thread) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = tid * work_load_per_thread;
    size_t e = (tid + 1) * work_load_per_thread;
    e = (e <= sz) ? e : sz;

    for (size_t i = b; i < e; ++i) {
        keys[i] *= 17;
    }
}


//  1073741824 -> 1073741789
//  65536 -> 65521

__global__ void test() {
    printf("test\n");
}

int main(int argc, char const *argv[])
{
    // size_t tm = std::numeric_limits<T>::max();
    // size_t sz = 1073741789;//65521;
    // size_t ptm = (tm + 1) / nthreads;

    // std::cout << "tm: \t" << tm << std::endl;
    // std::cout << "ptm: \t" << ptm << std::endl;

    T *keys;
    size_t keys_sz = 128 * 1024 *1024;
    size_t insert_loop = 10;
    // size_t keys_sz = 128;
    // size_t insert_loop = 1;

    checkKernelErrors(cudaMalloc((void **)&keys, sizeof(T) * keys_sz));

    T *seed;
    T seed_num = std::max(hash_table_num, WARP_SIZE);
    checkKernelErrors(cudaMalloc((void **)&seed, sizeof(T) * s_sz * seed_num));
    {
        char seed_buf[128];
        std::ifstream ifs("/dev/urandom");
        ifs.read(seed_buf, 128);
        std::seed_seq sq(seed_buf, seed_buf + 128);
        std::default_random_engine gen(sq);
        
        {
            // keys
            T *tk = new T[keys_sz];
            std::uniform_int_distribution<T> dis(0, std::numeric_limits<T>::max());

            for (int i = 0; i < keys_sz; ++i)
            {
                tk[i] = (dis(gen));
            }

            checkKernelErrors(cudaMemcpy(keys, tk, sizeof(T) * keys_sz, cudaMemcpyHostToDevice));

            delete [] tk;

        }

        {
            // seed
            T *ts = new T[s_sz * seed_num];
            std::uniform_int_distribution<T> dis(1, std::numeric_limits<T>::max());

            for (int i = 0; i < s_sz * seed_num; ++i)
            {
                ts[i] = (dis(gen));
                // std::cout << ts[i] << ",";
            }
            // std::cout << std::endl;

            checkKernelErrors(cudaMemcpy(seed, ts, sizeof(T) * s_sz * seed_num, cudaMemcpyHostToDevice));

            delete [] ts;
        }
        
       
    }

    {
        // debug
    }

    C *hash_table;
    checkKernelErrors(cudaMalloc((void **)&hash_table, sizeof(T) * hash_table_sz * hash_table_num));
    checkKernelErrors(cudaMemset(hash_table, 0, sizeof(T) * hash_table_sz * hash_table_num));

    C *count;
    checkKernelErrors(cudaMalloc((void **)&count, sizeof(C) * keys_sz));
    checkKernelErrors(cudaMemset(count, 0, sizeof(C) * keys_sz));

    
    // checkKernelErrors((test<<<1, 1>>>()));
// test<<<1, 1>>>();

    // {
    //     size_t work_load_per_thread = (keys_sz + (nthreads - 1)) / nthreads;
    //     std::cout << "work_load_per_thread: " << work_load_per_thread << std::endl;

    //     std::cout << "p1" << std::endl;
    //     using namespace std::chrono;
    //     high_resolution_clock::time_point t1 = high_resolution_clock::now();
        
    //     checkKernelErrors((insert<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_thread, hash_table, hash_table_num, hash_table_sz, seed, s_sz, nullptr)));

    //     for (size_t l = 1; l < insert_loop; ++l) {
    //         checkKernelErrors((random<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_thread)));
    //         checkKernelErrors((insert<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_thread, hash_table, hash_table_num, hash_table_sz, seed, s_sz, nullptr)));
    //     }
    //     cudaDeviceSynchronize();

    //     high_resolution_clock::time_point t2 = high_resolution_clock::now();
    //     duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    //     size_t total_bytes = sizeof(T) * keys_sz * hash_table_num * insert_loop;
    //     // std::cout << "total_bytes: " << total_bytes << "-->" << (total_bytes / (1ul<<30)) << std::endl;
    //     std::cout << "perf: " << double(total_bytes) / time_span.count() / (1ul<<30) << "GB/s" << std::endl;

    // }

    {
        size_t work_load_per_thread = (keys_sz + (nthreads - 1)) / nthreads;
        std::cout << "work_load_per_thread: " << work_load_per_thread << std::endl;

        size_t work_load_per_warp = (keys_sz + (nwarps - 1)) / nwarps;
        std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;

        std::cout << "p1" << std::endl;
        using namespace std::chrono;
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        
        checkKernelErrors((insert2<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp, hash_table, hash_table_num, hash_table_sz, seed, s_sz, nullptr)));

        for (size_t l = 1; l < insert_loop; ++l) {
            checkKernelErrors((random<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_thread)));
            checkKernelErrors((insert2<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp, hash_table, hash_table_num, hash_table_sz, seed, s_sz, nullptr)));
        }
        cudaDeviceSynchronize();

        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        size_t total_bytes = sizeof(T) * keys_sz * hash_table_num * insert_loop;
        // std::cout << "total_bytes: " << total_bytes << "-->" << (total_bytes / (1ul<<30)) << std::endl;
        std::cout << "perf: " << double(total_bytes) / time_span.count() / (1ul<<30) << "GB/s" << std::endl;

    }

        {
        size_t work_load_per_thread = (keys_sz + (nthreads - 1)) / nthreads;
        std::cout << "work_load_per_thread: " << work_load_per_thread << std::endl;

        size_t work_load_per_warp = (keys_sz + (nwarps - 1)) / nwarps;
        std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;

        std::cout << "p1" << std::endl;
        using namespace std::chrono;
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        
        checkKernelErrors((insert4<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp, hash_table, hash_table_num * hash_table_sz / 256, 256, seed, s_sz, nullptr)));

        for (size_t l = 1; l < insert_loop; ++l) {
            checkKernelErrors((random<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_thread)));
            checkKernelErrors((insert4<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp, hash_table, hash_table_num * hash_table_sz / 256, 256, seed, s_sz, nullptr)));
        }
        cudaDeviceSynchronize();

        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        size_t total_bytes = sizeof(T) * keys_sz * hash_table_num * insert_loop;
        // std::cout << "total_bytes: " << total_bytes << "-->" << (total_bytes / (1ul<<30)) << std::endl;
        std::cout << "perf: " << double(total_bytes) / time_span.count() / (1ul<<30) << "GB/s" << std::endl;

    }


    


    // std::cout << "p2" << std::endl;

    // checkKernelErrors((search<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_thread, hash_table, hash_table_num, hash_table_sz, seed, s_sz, count,nullptr)));

    // std::cout << "p3" << std::endl;


    // C *count_h = new C[keys_sz];
    // cudaMemcpy(count_h, count, sizeof(C) * keys_sz, cudaMemcpyDeviceToHost);

    // for (size_t i = 0; i < keys_sz; ++i) {
    //     std::cout << count_h[i] << ", ";
    // }
    // std::cout << std::endl;

    // cudaDeviceReset();

    return 0;
}