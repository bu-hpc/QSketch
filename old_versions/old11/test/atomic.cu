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

#define WORKLOAD_PER_WARP 102400
#define LIMIT 32
#define WORKLOAD_PER_THREAD (WORKLOAD_PER_WARP / LIMIT)


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



__global__ void test(unsigned int *c) {
    size_t tid = threadIdx.x;
    size_t gtid = (blockIdx.x * blockDim.x + threadIdx.x);
    for (size_t i = 0; i < WORKLOAD_PER_THREAD; ++i) {
        if (tid < LIMIT) {
            atomicAdd(c + gtid, 1);
        }
    }
}


int main(int argc, char const *argv[])
{
    int nblocks = 65536;
    int threads_per_block = 32;
    int nthreads = nblocks * threads_per_block;
    unsigned int *c;
    cudaMalloc(&c, sizeof(unsigned int) * nthreads);
    
    double ts = get_run_time([&](){
        test<<<nblocks, threads_per_block>>>(c);
        cudaDeviceSynchronize();
    });
    double workload = double(WORKLOAD_PER_WARP) * nblocks;
    // double workload = double(WORKLOAD_PER_THREAD) * nthreads;
    // std::cout << WORKLOAD_PERTHREAD << std::endl;
    // std::cout << nthreads << std::endl;
    std::cout << ts << std::endl;
    std::cout << "perf: " << format_perf(workload / ts) << std::endl;

    return 0;
}