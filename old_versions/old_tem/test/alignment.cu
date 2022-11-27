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

template <typename T>
__global__ void trans(T *keys, size_t sz, size_t work_load_per_warp,
    unsigned int *buf, size_t buf_sz) {
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;
    if (e >= sz) {
        e = sz;
    }

    for (size_t i = b; i < e; i += WARP_SIZE) {
        T v = keys[i + tid] % buf_sz;
        atomicAdd(buf + v, 1);
        atomicAdd(buf + v + 3, 1);
        atomicAdd(buf + v + 6, 1);
        // if (v & )
        // atomicAdd(buf + (v % buf_sz), 1);
        // atomicAdd(buf + (v % buf_sz) + (v % 7), 1);
        // atomicAdd(buf + (v % buf_sz) + (v % 3), 1);
        // atomicAdd(buf + ((v * 1000000007) % buf_sz), 1);
        // atomicAdd(buf + ((v * 65521) % buf_sz), 1);
        // atomicAdd(buf + (i % buf_sz), 1);
    }
}


template <typename T>
__global__ void pre_trans(T *keys, size_t sz, size_t work_load_per_warp,
    T *target_keys) {
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;
    if (e >= sz) {
        e = sz;
    }

    __shared__ unsigned long long id;

    if (tid == 0) {
        id = b;
    }
    __threadfence_block();

    for (size_t j = 0; j < 32; ++j) {
        for (size_t i = b; i < e; i += WARP_SIZE) {
            T v = keys[i + tid] % sz;
            T t = v / (4 * 1024 * 1024);
            if ((t == j)) {
                if (keys[i + tid] == 4267365876ul) {
                    printf("j: %lu\n", j);
                }
                size_t old = atomicAdd(&id, 1ul);
                target_keys[old] = keys[i + tid];
            }
        }
    }

    __threadfence_block();
    if (tid == 0) {
        if (id != e) {
            printf("err: %lu, %lu, %lu, diff: %lu, example: %u\n", wid, id, e, e - id, keys[b]);
        }
    }

    for (size_t i = b; i < e; i += WARP_SIZE) {
        keys[i + tid] = target_keys[i + tid];
    }
}

template <typename T>
__global__ void trans_loop(T *keys, size_t sz, size_t work_load_per_warp,
    unsigned int *buf, size_t buf_sz) {
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;
    if (e >= sz) {
        e = sz;
    }

    for (size_t j = 0; j < 32; ++j) {
        for (size_t i = b; i < e; i += WARP_SIZE) {
            T v = keys[i + tid] % buf_sz;
            T t = v / (4 * 1024 * 1024);
            if (t == j) {
                atomicAdd(buf + v, 1);
                atomicAdd(buf + v + 3, 1);
                atomicAdd(buf + v + 6, 1);
                // atomicAdd(buf + (v % buf_sz) + (v % 7), 1);
                // atomicAdd(buf + (v % buf_sz) + (v % 3), 1);
                // atomicAdd(buf + ((v * 1000000007) % buf_sz), 1);
                // atomicAdd(buf + ((v * 65521) % buf_sz), 1);
            }
        }
    } 
}

#define LOCAL_BUF_SZ 64

template <typename T>
__global__ void trans_loop_cal1(T *keys, size_t sz, size_t work_load_per_warp,
    T *buf, size_t buf_sz, size_t *b, size_t *m, size_t *e) {

    // size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    // size_t tid = threadIdx.x % WARP_SIZE;
    // size_t b = wid * work_load_per_warp;
    // size_t e = (wid + 1) * work_load_per_warp;


    // if (e >= sz) {
    //     e = sz;
    // }

    // __shared__ T local_buf[LOCAL_BUF_SZ];
    // __shared__ unsigned int local_buf_sz;

    // if (tid == 0) {
    //     local_buf_sz = 0;
    // }

    // __threadfence_block();

    // for (size_t j = 1; j <= 32; ++j) {
    //     for (size_t i = b; i < e; i += WARP_SIZE) {

    //         __threadfence_block();
    //         if (local_buf_sz >= WARP_SIZE) {
    //             // size_t old_global_id = atomicAdd(global_id, 32);
    //             size_t old_e = atomicAdd(e, WARP_SIZE);
    //             while ((old_e + buf_sz - __ldcv(b)) % buf_sz <= buf_sz - WARP_SIZE) {

    //             }

    //         }


    //         T v = keys[i + tid] % buf_sz;
    //         T t = v / (4 * 1024 * 1024);
    //         if ((t > j - 1) && (t < j)) {
    //             // atomicAdd(buf + (v % buf_sz), 1);
    //             unsigned int old = atomicAdd(&local_buf_sz, 1);
    //             local_buf[old] = v;
    //         }
    //     }
    // } 
}

template <typename T>
__global__ void trans_loop_cal2(T *keys, size_t sz, size_t work_load_per_warp,
    unsigned int *buf, size_t buf_sz) {
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;
    if (e >= sz) {
        e = sz;
    }

    for (size_t j = 1; j <= 32; ++j) {
        for (size_t i = b; i < e; i += WARP_SIZE) {
            T v = keys[i + tid] % buf_sz;
            T t = v / (4 * 1024 * 1024);
            if ((t > j - 1) && (t < j)) {
                atomicAdd(buf + (v % buf_sz), 1);
                // atomicAdd(buf + (v % buf_sz) + (v % 7), 1);
                // atomicAdd(buf + (v % buf_sz) + (v % 3), 1);
                // atomicAdd(buf + ((v * 1000000007) % buf_sz), 1);
                // atomicAdd(buf + ((v * 65521) % buf_sz), 1);
            }
        }
    } 
}


__global__ void add() {
    __shared__ unsigned int sz;
    size_t tid = threadIdx.x;
    if (tid == 0) {
        sz = 0;
    }

    unsigned int old = atomicAdd(&sz, 1);
    printf("tid: %lu, %u, %u\n", tid, old, sz);
}

template <typename T>
std::string cal_size(size_t sz) {
    double total_bytes = sizeof(T) * sz;
    if (total_bytes < (1ul << 10)) {
        return std::to_string(total_bytes / (1ul)) + " B";
    }
    if (total_bytes < (1ul << 20)) {
        return std::to_string(total_bytes / (1ul << 10)) + " KB";
    }
    if (total_bytes < (1ul << 30)) {
        return std::to_string(total_bytes / (1ul << 20)) + " MB";
    }
    
    return std::to_string(total_bytes / (1ul << 30)) + " GB";
}

// int main(int argc, char const *argv[])
// {
//     add<<<1, 32>>>();
//     cudaDeviceSynchronize();
//     return 0;
// }

int main(int argc, char const *argv[])
{
    size_t fifo_sz;
    cudaDeviceGetLimit(&fifo_sz,cudaLimitPrintfFifoSize);
    // std::cout << fifo_sz << std::endl;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, fifo_sz * 8);
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, 
            CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetGeneratorOffset(gen, 1024));


    unsigned int *keys;
    unsigned int *buf;
    // size_t keys_sz = 1024ul * 1024 * 1024;
    // size_t buf_sz = 1024ul * 1024 * 1024;
    size_t keys_sz = 128ul * 1024 * 1024;
    size_t buf_sz = 128ul * 1024 * 1024;
    checkKernelErrors(cudaMalloc(&keys, sizeof(unsigned int) * keys_sz));
    checkKernelErrors(cudaMalloc(&buf, sizeof(unsigned int) * buf_sz));
    CURAND_CALL(curandGenerate(gen, keys, keys_sz));

    unsigned int *hbuf = new unsigned int[buf_sz];
    memset(hbuf, 0, sizeof(unsigned int) * buf_sz);
    checkKernelErrors(cudaMemcpy(buf, hbuf, sizeof(unsigned int) * buf_sz, cudaMemcpyHostToDevice));

    std::cout << "keys: " << cal_size<unsigned int>(keys_sz) << std::endl;
    std::cout << "buf: " << cal_size<unsigned int>(buf_sz) << std::endl;

    int nblocks = 65536;
    int threads_per_block = 32;
    int nthreads = nblocks * threads_per_block;
    double workload = keys_sz;
    size_t work_load_per_warp = ceil<size_t>(keys_sz, nblocks);
        
    std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;

    if (true) {
        // size_t buf_sz_i = 128ul * 1024 * 1024;
        for (size_t buf_sz_i = 1024 * 1024; buf_sz_i <= buf_sz; buf_sz_i *= 2) 
        {
            memset(hbuf, 0, sizeof(unsigned int) * buf_sz);
            checkKernelErrors(cudaMemcpy(buf, hbuf, sizeof(unsigned int) * buf_sz, cudaMemcpyHostToDevice));
            double ts = get_run_time([&](){
                trans<unsigned int><<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp, buf, buf_sz_i);
                cudaDeviceSynchronize();
            });
            std::cout << (buf_sz_i / 1024 / 1024) << " : " << ts;
            std::cout << "\tperf: " << format_perf(workload / ts) << std::endl;

            checkKernelErrors(cudaMemcpy(hbuf, buf, sizeof(unsigned int) * buf_sz, cudaMemcpyDeviceToHost));

            std::cout << "hbuf 1007: " << hbuf[1007] << std::endl;
        }
    }

    {
        std::cout << "--------------------------example :---------------------------" << std::endl;
        size_t buf_sz = 128ul * 1024 * 1024;
        double ts = get_run_time([&](){
            trans_loop<unsigned int><<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp, buf, buf_sz);
            cudaDeviceSynchronize();
        });
        std::cout << ts;;
        std::cout << "\tperf: " << format_perf(workload / ts) << std::endl;
    }

    {
        std::cout << "--------------------------target_keys :---------------------------" << std::endl;
        unsigned int *target_keys;
        checkKernelErrors(cudaMalloc(&target_keys, sizeof(unsigned int) * keys_sz));
        pre_trans<unsigned int><<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp, target_keys);

        double ts = get_run_time([&](){
            trans<unsigned int><<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp, buf, buf_sz);
            cudaDeviceSynchronize();
        });
        std::cout << ts;;
        std::cout << "\tperf: " << format_perf(workload / ts) << std::endl;
    }

    

    return 0;
}


int main2(int argc, char const *argv[])
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