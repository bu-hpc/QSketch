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
#include <unordered_map>
#include <unordered_set>

#include <cuda_runtime.h>
#include <curand.h>

__global__ void test() {
    size_t tid = threadIdx.x;
    size_t sub_warp_id = tid / 8;

    unsigned int mask = 0xff << (sub_warp_id * 8);
    unsigned int val = tid;
    val = __shfl_sync(mask, val, sub_warp_id * 8);
    printf("tid: %lu, %u\n", tid, val);

    // unsigned int m = 0;
    // if (tid < 16) {
    //     m = __activemask();
    // }

    // printf("tid: %lu, m: %u\n", tid, m);
}

#define SUB_WARP_SIZE 8


__global__ void sub_warp_min() {
    size_t tid = threadIdx.x;
    unsigned char sub_warp_id = tid / SUB_WARP_SIZE;
    // __shared__ int buf[32];
    // buf[tid] = tid;
    unsigned int thread_min_low = 32 - tid;

    for (int j = 4; j >= 1; j = j >> 1) {
        unsigned int t_low = __shfl_down_sync(0xffffffff, thread_min_low, j);
        if ((tid >= sub_warp_id * SUB_WARP_SIZE) && (tid < sub_warp_id * SUB_WARP_SIZE + j)) {
            thread_min_low = min(thread_min_low, t_low);
        }
    }
    // if (tid % SUB_WARP_SIZE == 0) {
    //     count[tid / SUB_WARP_SIZE] = thread_min_low;
    // }
    printf("%lu, %u\n", tid, thread_min_low);
}

__global__ void any() {
    size_t tid = threadIdx.x;
    unsigned char sub_warp_id = tid / SUB_WARP_SIZE;
    unsigned int sub_warp_mask = 0xff << (sub_warp_id * 8);

    unsigned int max_count = 1;
    if (tid == 6) {
        max_count = 150;
    }

    if (tid == 26) {
        max_count = 150;
    }

    printf("tid: %lu, max_count: %u\n",tid, max_count);

    int update_low = __any_sync(sub_warp_mask, max_count > 128);

    printf("tid: %lu, update_low: %d\n",tid, update_low);
}

int main(int argc, char const *argv[])
{
    // test<<<1, 32>>>();
    // sub_warp_min<<<1, 32>>>();
    any<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}