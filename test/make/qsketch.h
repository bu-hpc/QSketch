#pragma once

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

namespace qsketch {

using uint = unsigned int;

// global variables
extern const bool DEBUG_MODE;
extern const bool RANDOM_SEED;

__global__ void test_kernel();

struct Test_Struct
{
    int a;
};

}