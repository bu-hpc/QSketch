
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

namespace Test_Kernel {
    __global__ void k();
    __global__ void g();
}


class Test
{
public:
    virtual void run(char b) {
        using namespace Test_Kernel;
        std::cout << "Test run" << std::endl;
        if (b == 'k') {
            k<<<1, 32>>>();
        } else if (b == 'g') {
            g<<<1, 32>>>();
        }
    }
};

namespace Test_Kernel {
__global__ void k() {
    size_t tid = threadIdx.x;
    if (tid == 0) {
        printf("Test k\n");
    }
}

__global__ void g() {
    size_t tid = threadIdx.x;
    if (tid == 0) {
        printf("Test g\n");
    }
}

}


// class D_Test : Test {
// public:
//     __global__ void k() {
//         size_t tid = threadIdx.x;
//         if (tid == 0) {
//             printf("D_Test k\n");
//         }
//     }

//     __global__ void g() {
//         size_t tid = threadIdx.x;
//         if (tid == 0) {
//             printf("D_Test g\n");
//         }
//     }

//     void run(char b) {
//         std::cout << "D_Test run" << std::endl;
//         if (b == 'k') {
//             k<<<1, 32>>>();
//         } else if (b == 'g') {
//             g<<<1, 32>>>();
//         }
//     }
// };

int main(int argc, char const *argv[])
{
    Test t;
    t.run('g');

    // D_Test dt;
    // Test *ptr = &dt;
    // dt.run('k');
    // ptr->run('g');
    cudaDeviceSynchronize();
    return 0;
}