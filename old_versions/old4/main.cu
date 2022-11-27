#include "lib/lib.h"

// std::unordered_set<void *> pointers;

constexpr size_t default_hash_table_sz = 65521;
constexpr size_t default_hash_table_num = 5;
constexpr size_t default_seed_sz = 2;
constexpr size_t defualt_seed_num = default_hash_table_num;


constexpr size_t default_hash_n = 65521;
constexpr size_t default_hash_m = 5;

const char* curandGetStatusString(curandStatus_t status) {
// detail info come from http://docs.nvidia.com/cuda/curand/group__HOST.html
    switch(status) {
        case CURAND_STATUS_SUCCESS:                     return "CURAND_STATUS_SUCCESS";
        case CURAND_STATUS_VERSION_MISMATCH:            return "CURAND_STATUS_VERSION_MISMATCH";
        case CURAND_STATUS_NOT_INITIALIZED:             return "CURAND_STATUS_NOT_INITIALIZED";
        case CURAND_STATUS_ALLOCATION_FAILED:           return "CURAND_STATUS_ALLOCATION_FAILED";
        case CURAND_STATUS_TYPE_ERROR:                  return "CURAND_STATUS_TYPE_ERROR";
        case CURAND_STATUS_OUT_OF_RANGE:                return "CURAND_STATUS_OUT_OF_RANGE";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:         return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:   return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        case CURAND_STATUS_LAUNCH_FAILURE:              return "CURAND_STATUS_LAUNCH_FAILURE";
        case CURAND_STATUS_PREEXISTING_FAILURE:         return "CURAND_STATUS_PREEXISTING_FAILURE";
        case CURAND_STATUS_INITIALIZATION_FAILED:       return "CURAND_STATUS_INITIALIZATION_FAILED";
        case CURAND_STATUS_ARCH_MISMATCH:               return "CURAND_STATUS_ARCH_MISMATCH";
        case CURAND_STATUS_INTERNAL_ERROR:              return "CURAND_STATUS_INTERNAL_ERROR";
    }
    return "CURAND_STATUS_UNKNOWN_ERROR";
}

int main(int argc, char const *argv[])
{
    // cudaDeviceReset();
    size_t insert_keys_size = 32 * 1024 * 1024;
    // size_t search_keys_size = 1024;

    using Key_T = unsigned int; 
    using Count_T = unsigned int;
    using Hashed_T = unsigned int;
    using Seed_T = unsigned int;


    Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T> map;
    Unordered_Map<Key_T, Count_T> emap;
    Test<unsigned int, unsigned int> test(gpu_tool<Key_T>, (map), (emap));

    test.insert_perf_test(insert_keys_size, 3);
    test.clear();
    // test.search_perf_test(insert_keys_size, search_keys_size);
    // test.clear();
    // test.search_accuracy_test(insert_keys_size, search_keys_size);
    // map.print(debug_log);
    return 0;
}

int main1(int argc, char const *argv[])
{
    size_t insert_keys_size = 32 * 1024;
    size_t search_keys_size = 1024;

    using Key_T = unsigned int; 
    using Count_T = unsigned int;

    // CPU_Tools<Key_T> tool;
    // Unordered_Map<Key_T, Count_T> map;
    Count_Min_Sketch_CPU<Key_T, Count_T> map;
    Unordered_Map<Key_T, Count_T> emap;
    // Test<unsigned int, unsigned int> test(tool, static_cast<Sketch<Key_T, Count_T> &>(map), static_cast<Sketch<Key_T, Count_T> &>(emap));
    Test<unsigned int, unsigned int> test(cpu_tool<Key_T>, (map), (emap));

    // test.insert_perf_test(insert_keys_size, 3);
    // test.clear();
    // test.search_perf_test(insert_keys_size, search_keys_size);
    // test.clear();
    // test.search_accuracy_test(insert_keys_size, search_keys_size);
    // map.print(debug_log);
    return 0;
}