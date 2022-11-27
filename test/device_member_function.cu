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

template <typename T>
struct Test {
    T *data;
    __device__  T &get(size_t id) {
        return data[id];
    }
    //  __host__ T &get(size_t id) {
    //     return data[id];
    // }
    __device__ __host__ T &set(size_t id, const int &val) {
        return data[id] = val;
    }
};

template <typename T>
__global__ void kernel(Test<T> t) {
    size_t tid = threadIdx.x;
    if (tid == 0) {
        t.set(21, 21);
        printf("device: %d\n", t.get(21));
    }
}

int main(int argc, char const *argv[])
{
    Test<int> th, td;
    th.data = new int[100];
    cudaMalloc(&td.data, sizeof(int) * 100);

    th.set(20, 20);
    std::cout << th.get(20) << std::endl;

    kernel<int><<<1, 32>>>(td);
    cudaDeviceSynchronize();
    return 0;
}