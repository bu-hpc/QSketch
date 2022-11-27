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


template<typename T>
__constant__ T cdata[CONSTANT_SIZE];

template<typename T>
__device__ T ddata[CONSTANT_SIZE];

template<typename T>
T data[CONSTANT_SIZE];

template<typename T>
__global__ void test(unsigned long long *ans) {
    size_t tid = threadIdx.x;
    size_t sum = 0;
    for (unsigned int i = 0; i < WORKLOAD_PERTHREAD; ++i)
    {
        unsigned int cid = ((i + 7) * 1000087 + tid * 1007) % CONSTANT_SIZE;
        sum += 1007 * (i + 7) * ddata<T>[cid] % 1000087;
    }
    // sum = cdata[tid];
    // printf("sum: %lu\n", sum);
    atomicAdd(ans, sum);

}
__device__ unsigned long long ans;

int main(int argc, char const *argv[])
{
    /* code */
    for (int i = 0; i < CONSTANT_SIZE; ++i) {
        // data[i] = ((i + 1007) * 100007) % 97;
        data<unsigned int>[i] = i;
    }
    cudaMemcpyToSymbol(cdata<unsigned int>, data<unsigned int>, sizeof(unsigned int) * CONSTANT_SIZE);
    cudaMemcpy(&ddata<unsigned int>, &data<unsigned int>, sizeof(unsigned int) * CONSTANT_SIZE, cudaMemcpyHostToDevice);
    int nblocks = 65536;
    int threads_per_block = 32;
    int nthreads = nblocks * threads_per_block;

    double ts = get_run_time([&](){
        test<unsigned int><<<nblocks, threads_per_block>>>(&ans);
        cudaDeviceSynchronize();
    });

    double workload = double(WORKLOAD_PERTHREAD) * nthreads;
    // std::cout << WORKLOAD_PERTHREAD << std::endl;
    // std::cout << nthreads << std::endl;
    std::cout << ts << std::endl;
    std::cout << "perf: " << format_perf(workload / ts) << std::endl;
    

    return 0;
}