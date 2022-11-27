#pragma once

#include <cassert>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <atomic>
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

namespace qsketch 
{

using uchar = unsigned char;
using uint = unsigned int;
using ulong = unsigned long;
using ullong = unsigned long long;

// global variables

/*
    DEBUG mode may print more detail information and disable some optimization. 
*/


// #define QSKETCH_DEBUG
#ifndef QSKETCH_DEBUG
    #define RELEASE
#endif
#define QSKETCH_ASYNC_MEMCPY
#define QSKETCH_ANALYZE




// enum Mode
// {
//     DEBUG,
//     RELEASE
// };

// extern const Mode mode;


/*
    1. It will try to generate the random seeds via reading /usr/urandom.
    2. If it failed to read /usr/urandom, it will generate the random seeds based on the system time. 
    3. If RANDOM_SEED equals false, 
*/
// extern const bool RANDOM_SEED;
#define RANDOM_SEED true


std::ostream &operator<<(std::ostream &os, const dim3 &d);
const char* curandGetStatusString(curandStatus_t status);

int read_file(const std::string &file_name, void **buf, size_t &sz);
int write_file(const std::string &file_name, void *buf, size_t sz);

int check_cuda_error();
size_t find_greatest_prime(size_t upper_bound);

#ifdef QSKETCH_DEBUG
    
    #define CUDA_CALL(expr)                                             \
        do {                                                            \
            expr;                                                       \
            cudaDeviceSynchronize();                                    \
            cudaError_t __err = cudaGetLastError();                     \
            if((__err) != cudaSuccess) {                                \
                printf("Error at %s:%d:'%s','%s'\n",__FILE__, __LINE__, \
                    #expr, cudaGetErrorString(__err));                  \
                abort();                                                \
            }                                                           \
        } while(0)

    #define CURAND_CALL(expr)                                           \
        do {                                                            \
            curandStatus_t __err = expr;                                \
            if((__err) != CURAND_STATUS_SUCCESS) {                      \
                printf("Error at %s:%d:'%s','%s'\n",__FILE__, __LINE__, \
                    #expr, curandGetStatusString(__err));               \
                abort();                                                \
            }                                                           \
        } while(0)

    extern size_t total_memory_usage;

#else

    #define CUDA_CALL(expr)                                             \
        expr;

    #define CURAND_CALL(expr)                                           \
        expr;


#endif

}


#ifdef QSKETCH_ANALYZE

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include "analyze/analyze.h"

#endif

#include "default.h"
#include "tool/tool.h"
#include "sketch/sketch.h"
#include "test/test.h"


