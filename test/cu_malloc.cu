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

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
// #include "analyze/analyze.h"

using Count_T = unsigned int;

void test() {
    size_t batch_sz = 1024;
    Count_T *counts = nullptr;
    Count_T *e_counts = nullptr;

    thrust::device_vector<Count_T> dc1(counts, counts + batch_sz);
    thrust::device_vector<Count_T> dc2(e_counts, e_counts + batch_sz);
    
    thrust::device_vector<Count_T> dc(dc1.size());
    thrust::transform(dc1.begin(), dc1.end(), dc2.begin(), dc.begin(), thrust::minus<Count_T>());


    // auto dc = diff(dvc, dvec);
    // std::cout << dc.size() << std::endl;
    Count_T max = thrust::reduce(dc.begin(), dc.end(), Count_T(0), thrust::maximum<Count_T>());
}

int main(int argc, char const *argv[])
{
    size_t keys_sz = 25165716;
    unsigned int *keys;
    std::cout << "p1" << std::endl;
    cudaMalloc(&keys, sizeof(unsigned int) * keys_sz);
    std::cout << "p2" << std::endl;
    cudaMemset(keys, 0, sizeof(unsigned int) * keys_sz);
    std::cout << "p3" << std::endl;
    cudaDeviceSynchronize();
    return 0;
}


