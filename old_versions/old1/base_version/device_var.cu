#include <cuda_runtime.h>
#include <iostream>

struct Test
{
    int d = 100;
};

__device__ Test t;


__global__ void test() {
    printf("test: %d\n", t.d);
}


int main(int argc, char const *argv[])
{
    test<<<1, 1>>>();
            cudaDeviceSynchronize();
    return 0;
}