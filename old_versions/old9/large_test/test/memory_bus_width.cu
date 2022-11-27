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


__global__ void test() {
    size_t tid = threadIdx.x;
    // bool val;s
    __shared__ int val[32];
    for (int i = 0; i <= 32; ++i) {
        val[tid] = i <= tid;
        int update_low = __any_sync(0xffffffff, i <= tid);
        if (tid == 0) {
            for (int j = 0; j < 32; ++j) {
                if (val[j]) {
                    printf("1");
                } else {
                    printf("0");
                }
            }
            printf("\n");
            printf("update_low: %d, %d\n",i, update_low);
        }
    }
}


int main(int argc, char const *argv[])
{

// __host__​cudaError_t cudaGetDeviceProperties ( cudaDeviceProp* prop, int  device )
// Returns information about the compute-device.
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << prop.memoryBusWidth << std::endl;
    return 0;
}