#include <iostream>
#include <cuda_runtime.h>

struct Hash
{
    int val;

    Hash() = default;

    __device__ __host__ Hash(int v) : val(v) {}

    __device__ __host__ int operator()(int key) {
        return key * val;
    }
};

__device__ Hash h(1);

__global__ void test() {
    size_t tid = (blockIdx.x * blockDim.x + threadIdx.x);
    // Hash h(tid);
    printf("%lu : %d\n", tid, h(tid));
}

int main(int argc, char const *argv[])
{
    test<<<1, 32>>>();
        cudaDeviceSynchronize();

    return 0;
}
