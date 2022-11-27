#include <cassert>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <bitset>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include <cuda_runtime.h>
#include <curand.h>



namespace test {

template <typename T>
struct Test
{
    virtual void run() {
        std::cout << "base run" << std::endl;
    }
};

// namespace Test_kernel1_kernel {
//     template<typename T>
//     __global__ void kernel_1() {
//         size_t tid = threadIdx.x;
//         if (tid == 0) {
//             printf("kernel_1\n");
//         }
//     }
// }

// template <typename T>
// struct Test_kernel1 : Test<T>
// { 
//     virtual void run() {
//         using namespace Test_kernel1_kernel;
//         Test_kernel1_kernel::kernel_1<T><<<1, 32>>>();
//         // kernel_1<T><<<1, 32>>>();
//         cudaDeviceSynchronize();
//     }
// };

// #define CLASS_NAME_KERNEL(n) n##_kernel

#define CLASS_NAME Test_kernel1
#define CLASS_NAME_KERNEL Test_kernel1_kernel

namespace CLASS_NAME_KERNEL {
    template<typename T>
    __global__ void kernel() {
        size_t tid = threadIdx.x;
        if (tid == 0) {
            printf("kernel_1\n");
        }
    }
}

template <typename T>
struct CLASS_NAME : Test<T>
{ 
    virtual void run() {
        using namespace CLASS_NAME_KERNEL;
        kernel<T><<<1, 32>>>();
        // Test_kernel1_kernel::kernel_1<T><<<1, 32>>>();
        cudaDeviceSynchronize();
    }
};
#undef CLASS_NAME
#undef CLASS_NAME_KERNEL


#define CLASS_NAME Test_kernel2
#define CLASS_NAME_KERNEL Test_kernel2_kernel

namespace CLASS_NAME_KERNEL {
    template<typename T>
    __global__ void kernel() {
        size_t tid = threadIdx.x;
        if (tid == 0) {
            printf("kernel_2\n");
        }
    }
}

template <typename T>
struct CLASS_NAME : Test<T>
{ 
    virtual void run() {
        using namespace CLASS_NAME_KERNEL;
        kernel<T><<<1, 32>>>();


        // Test_kernel1_kernel::kernel_1<T><<<1, 32>>>();
        cudaDeviceSynchronize();
    }
};
#undef CLASS_NAME
#undef CLASS_NAME_KERNEL

}