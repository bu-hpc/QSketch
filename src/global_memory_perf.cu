#include <qsketch.h>

#define PRIME 131071

// using Key_T = unsigned int; 
using Key_T = float;
using uint = unsigned int;

template <typename T>
__global__ void random_access(T *src, size_t sz, T *dec, int cmas) {
    // cmas : coalesced memory access size
    uint xid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (xid == 0) {
    //     printf("sz: %u\n", sz);
    // }
    xid = (xid / cmas) * cmas;
    xid = xid * PRIME % sz;
    // if (xid >= sz) {
    //     printf("%u\n", xid);
    // }
    dec[xid] = src[xid];
}

template <typename T>
__global__ void sequential_access(T *src, size_t sz, T *dec, int cmas) {
    uint xid = blockIdx.x * blockDim.x + threadIdx.x;
    uint nxid = (xid / cmas) * cmas;
    nxid = xid * PRIME % sz;
    dec[xid] = src[xid];
}

template <typename T>
__global__ void stride_copy(T *src, size_t sz, T *dec, int stride) {
    uint xid = blockIdx.x * blockDim.x + threadIdx.x;
    // uint nxid = (xid / cmas) * cmas;
    // nxid = xid * PRIME % sz;
    xid *= stride;
    dec[xid] = src[xid];    
}

int main(int argc, char const *argv[])
{
    Key_T *src = nullptr;
    Key_T *dec = nullptr;
    size_t sz = 1024 * 1024 * 1024;
    size_t nblock = sz / 32 / 32;
    const size_t work_load = sz / 32;

    std::cout << "work_load: " << qsketch::format_memory_usage<Key_T>(sz) << std::endl;

    // src = qsketch::curand_gpu_tool<Key_T>.zero(src, sz);
    // dec = qsketch::curand_gpu_tool<Key_T>.zero(dec, sz);
    cudaMalloc(&src, sizeof(Key_T) * sz);
    cudaMalloc(&dec, sizeof(Key_T) * sz);
    // {
    //     double tts = qsketch::get_run_time([&]() {
    //         // map.search(keys, batch_sz, counts);
    //         random_access<Key_T><<<nblock, 32>>>(src, sz, dec, 1);
    //         cudaDeviceSynchronize();
    //     });
    //     // format_bandwidth
    //     cudaError_t __err = cudaGetLastError();                     \
    //     if((__err) != cudaSuccess) {                                \
    //         printf("Error at %s:%d:'%s','%s'\n",__FILE__, __LINE__, \
    //             "random_access", cudaGetErrorString(__err));                  \
    //         abort();                                                \
    //     }     
    //     std::cout << "perf: " << qsketch::format_bandwidth<double, Key_T>(work_load / tts) << std::endl;
    // }
    
    // {
    //     double tts = qsketch::get_run_time([&]() {
    //         // map.search(keys, batch_sz, counts);
    //         sequential_access<Key_T><<<nblock, 32>>>(src, sz, dec, 1);
    //         cudaDeviceSynchronize();
    //     });
    //     // format_bandwidth
    //     cudaError_t __err = cudaGetLastError();                     \
    //     if((__err) != cudaSuccess) {                                \
    //         printf("Error at %s:%d:'%s','%s'\n",__FILE__, __LINE__, \
    //             "sequential_access", cudaGetErrorString(__err));                  \
    //         abort();                                                \
    //     }   
    //     std::cout << "perf: " << qsketch::format_bandwidth<double, Key_T>(work_load / tts) << std::endl;
    
    // }

    for (int stride = 1; stride <= 32; stride++) {
        double tts = qsketch::get_run_time([&]() {
            // map.search(keys, batch_sz, counts);
            stride_copy<Key_T><<<nblock, 32>>>(src, sz, dec, stride);
            cudaDeviceSynchronize();
        });
        // format_bandwidth
        cudaError_t __err = cudaGetLastError();                     \
        if((__err) != cudaSuccess) {                                \
            printf("Error at %s:%d:'%s','%s'\n",__FILE__, __LINE__, \
                "stride_copy", cudaGetErrorString(__err));                  \
            abort();                                                \
        }   
        // std::cout << "stride: " << stride << "\tperf: " << qsketch::format_bandwidth<double, Key_T>(work_load / tts) << std::endl;
        // double time_s = elapsedTimeInMs / 1e3;
        double bandwidthInGBs = (2.0f * sizeof(Key_T) * work_load) / (double)1e9;
        bandwidthInGBs = bandwidthInGBs / tts;
        std::cout << bandwidthInGBs << std::endl;
    }
    
    // qsketch::curand_gpu_tool<Key_T>.free(src);
    // qsketch::curand_gpu_tool<Key_T>.free(dec);
    cudaFree(src);
    cudaFree(dec);
    return 0;
}