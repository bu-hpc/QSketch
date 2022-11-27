
#include "lib.cuh"




__global__ void kernel_2() {
    size_t tid = threadIdx.x;
    if (tid == 0) {
        printf("kernel_1\n");
    }
}

namespace test {
    // namespace Test_kernel1 {
    //     __global__ void kernel_1() {
    //         size_t tid = threadIdx.x;
    //         if (tid == 0) {
    //             printf("kernel_1\n");
    //         }
    //     }
    // }
// template <typename T>
// void Test_kernel1<T>::run() {
//     kernel_1<<<1, 32>>>();
//     cudaDeviceSynchronize();
// }
}

// test::Test_kernel1<int> a;