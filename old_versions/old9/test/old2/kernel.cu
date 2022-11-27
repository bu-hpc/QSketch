
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

#include <cuda_runtime.h>
#include <curand.h>

__global__ void test(int *val) {
    if (threadIdx.x == 0) {
        printf("i\n");
        atomicAdd(val, 1);
    }
}

int main(int argc, char const *argv[])
{

    size_t fifo_sz;
    cudaDeviceGetLimit(&fifo_sz,cudaLimitPrintfFifoSize);
    // std::cout << fifo_sz << std::endl;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, fifo_sz * 8);

    int nb = atoi(argv[1]); 
    int nt = atoi(argv[2]);
    int hval;
    int *dval;
    cudaMalloc(&dval, sizeof(int));
    cudaMemset(dval, 0, sizeof(int));
    test<<<nb, nt>>>(dval);
    cudaMemcpy(&hval, dval, sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << hval << std::endl;
    // while (1) {
    //     std::cerr << "l" << std::endl;
    // }

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}