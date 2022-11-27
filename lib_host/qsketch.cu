#include <qsketch.h>

namespace qsketch {

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

std::ostream &operator<<(std::ostream &os, const dim3 &d) {
    return os << d.x << ", " << d.y << ", " << d.z;
}

    namespace calculator {

        size_t number_of_buckets(size_t insert_keys_sz, size_t m, double factor) // default value of factor = 1.0
        {
            size_t table_sz = insert_keys_sz / factor;

            std::cerr << (table_sz / m) << std::endl;
            return table_sz / m;
        }
    }

int check_cuda_error() {
    cudaError_t __err = cudaGetLastError();                     
    if((__err) != cudaSuccess) {                                
        printf("Error at %s:%d:'%s'\n",__FILE__, __LINE__, 
            cudaGetErrorString(__err));                  
        abort();                                                
    }
    return 0;
}

size_t find_greatest_prime(size_t upper_bound) {
    static size_t ps[] = {
        402653171,
        268435399,
        201326557,
        134217689,
        100663291,
        67108859,
        50331599,
        33554393,
        25165813,
        16777213,
        12582893,
        8388593,
        6291449,
        4194301,
        3145721,
        2097143,
        1572853,
        1048573,
        786431,
        524287,
        393209,
        262139,
        196597,
        131071,
        98299,
        65521,
        49139,
        32749,
        24571,
        16381,
        12281,
        8191,
        6143,
        4093,
        3067,
        2039,
        1531,
        1021,
        761,
        509,
        383,
        251,
        191,
        127,
        89,
        61,
        47,
        31,
        23,
        13,
        11,
        7,
        5,
        3,
        2
    };
    size_t ps_sz = sizeof(ps) / sizeof(size_t);
    for (int i = 0; i < ps_sz; ++i) {
        if (ps[i] <= upper_bound) {
            return ps[i];
        }
    }
    return 0;
}


#ifdef QSKETCH_DEBUG

    size_t total_memory_usage = 0;

#endif
}