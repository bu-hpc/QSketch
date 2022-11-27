#include <cuda_runtime.h>

#include <iostream>

__global__ void kernel_1(size_t sz, int *c) {
    for (size_t i = 0; i < sz; ++i) {
        int v = atomicAdd(c, 1) + 1;
        atomicMax(c + 1, v);
        atomicMin(c + 2, v);
    }
    // atomicSub(d, 1);
}


__global__ void kernel_2(size_t sz, int *c) {
    for (size_t i = 0; i < sz; ++i) {
        int v = atomicSub(c, 1) - 1;
        atomicMax(c + 1, v);
        atomicMin(c + 2, v);
    }
}

int no_overlap()
{
    int *c;
    cudaMalloc(&c, sizeof(int) * 3);
    int hc[3] = {0, 0, 0};

    
    kernel_1<<<32, 32>>>(1024, c);
    kernel_2<<<32, 32>>>(1024, c);

    cudaMemcpy(&hc, c, sizeof(int) * 3, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < 3; ++i) {
        std::cout << hc[i] << std::endl;
    }

    return 0;
}


int main(int argc, char const *argv[])
{
    int *c;
    cudaMalloc(&c, sizeof(int) * 3);
    int hc[3] = {0, 0, 0};

    cudaStream_t stream[2];
    for (size_t i = 0; i < 2; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    kernel_1<<<32, 32, 0, stream[0]>>>(1024, c);
    kernel_2<<<32, 32, 0, stream[1]>>>(1024, c);

    cudaDeviceSynchronize();
    cudaMemcpy(&hc, c, sizeof(int) * 3, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < 3; ++i) {
        std::cout << hc[i] << std::endl;
    }

    return 0;
}