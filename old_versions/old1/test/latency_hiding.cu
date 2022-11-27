#include <cuda_runtime.h>
#include <cuda.h>
// #include <crt/sm_80_rt.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <chrono>

#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)

// to do
// use 8bit unsigned char

#define WARP_SIZE 32ul

using T = unsigned int; // the type of key and mapped
using C = unsigned int; // the type of count
C C_MAX = std::numeric_limits<C>::max();

using ull = unsigned long long;

const size_t nblocks = 65536;
// const size_t nblocks = 1;
const size_t threads_per_block = 32;
const size_t nthreads = nblocks * threads_per_block;
const size_t nwarps = (nthreads + WARP_SIZE - 1) / WARP_SIZE;


// const size_t hash_table_sz = 65521;
// const size_t hash_table_num = 16;
const size_t n = 4093;
const size_t n_alignment = 4096;
const size_t m = 8;
// const size_t k = 32;

const size_t nseed = m * WARP_SIZE + 1;
const size_t s_sz = 2;

__device__ T hash(T *seed, T s_sz, const T &key) {
    T hv = key;
    for (size_t i = 0; i < s_sz; ++i)
    {
        hv = hv * seed[i];
    }
    return hv;
}

__device__ T hash2(T *seed, T s_sz, const T &key) {
    T hv = 0;
    for (size_t i = 0; i < s_sz; ++i)
    {
        hv += key * seed[i];
    }
    return hv;
}

__device__ T hash3(T *seed, T s_sz, const T &key) {
    T hv = 0;
    T k = key;
    for (size_t i = 0; i < s_sz; ++i)
    {
        hv += k * seed[i];
        k *= key;
    }
    return hv;
}

__device__ T hash4(T *seed, T *seed2, T s_sz, const T &key) {
    T hv = 0;
    for (size_t i = 0; i < s_sz; ++i)
    {
        hv += (key + seed2[i]) * seed[i];
    }
    return hv;
}


__global__ void insert(T *keys, size_t sz,  size_t work_load_per_warp,
    C *hash_table, size_t n, size_t m,
    T *seed, size_t s_sz, 
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    for (size_t i = b; i < e; i++) {
        __shared__ T *h;
        T v = keys[i];
        if (tid == 0) {
            h = hash_table + (hash2(seed, s_sz, v) % n) * m * WARP_SIZE;
        }

        for (size_t j = 0; j < m; ++j) {
            size_t mid = j * WARP_SIZE + tid;
            if (hash2(seed + mid + 1, s_sz, v) % (7) == 1) {
                atomicAdd(h + mid, 1);
            }
        }
    }
}

__global__ void insert2(T *keys, size_t sz,  size_t work_load_per_warp,
    C *hash_table, size_t n, size_t m,
    T *seed, size_t s_sz, 
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    __shared__ T v[WARP_SIZE];
    __shared__ T h[WARP_SIZE];

    for (size_t i = b; i < e; i += WARP_SIZE) {
        size_t t = i + tid;
        v[tid] = keys[t];
        h[tid] = (hash2(seed, s_sz, keys[t]) % n) * m * WARP_SIZE;

        for (size_t w = 0; w < WARP_SIZE; ++w) {
            for (size_t j = 0; j < m; ++j) {
                size_t mid = j * WARP_SIZE + tid;
                // h[w]++;
                v[w]++;

                // hash2(seed + mid + 1, s_sz, 101/*keys[t]*/);
                // atomicAdd(hash_table + h[w] + mid, 1);
                // if (hash2(seed + mid + 1, s_sz, keys[t]/*keys[t]*/) % (7) == 1) {
                //     // atomicAdd(hash_table + h[w] + mid, 1);
                // }
            }
        }
    }
}

__global__ void search(T *keys, size_t sz,  size_t work_load_per_warp,
    C *hash_table, size_t n, size_t m,
    T *seed, size_t s_sz, 
    C *count, C C_MAX,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    for (size_t i = b; i < e; i++ ) {
        // size_t t = i + tid;
        // printf("%lu\n", t);
        __shared__ T *h;

        // __shared__ T key_warp[WARP_SIZE];
        T v = keys[i];
        if (tid == 0) {
            h = hash_table + (hash2(seed, s_sz, v) % n) * m * WARP_SIZE;
        }

        __shared__ C min[WARP_SIZE];
        min[tid] = C_MAX;

        for (size_t j = 0; j < m; ++j) {
            size_t mid = j * WARP_SIZE + tid;
            if (hash2(seed + mid + 1, s_sz, v) % (7) == 1) {
                // atomicAdd(h[mid], 1);
                if (h[mid] < min[tid]) {
                    min[tid] = h[mid];
                }
            }
        }
        // __reduce_min_sync();
        // __activemask();
        // __reduce_max_sync(0xffffffff, 1);
        // T val = __reduce_min_sync(0xffffffff, min[tid]);

        // if (tid == 0) {
        //     count[i] = val;
        // } 

        C ans = C_MAX;
        if (tid == 0) {
            for (size_t j = 0; j < WARP_SIZE; ++j) {
                if (min[j] < ans) {
                    ans = min[j];
                }
            }
            count[i] = ans;
        }
    }
}

__global__ void search2(T *keys, size_t sz,  size_t work_load_per_warp,
    C *hash_table, size_t n, size_t m,
    T *seed, size_t s_sz, 
    C *count, C C_MAX,
    void *debug)
{
    size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;
    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;

    for (size_t i = b; i < e; i++ ) {
        // size_t t = i + tid;
        // printf("%lu\n", t);
        __shared__ T *h;

        // __shared__ T key_warp[WARP_SIZE];
        T v = keys[i];
        if (tid == 0) {
            h = hash_table + (hash2(seed, s_sz, v) % n) * m * WARP_SIZE;
        }

        __shared__ C min_values[WARP_SIZE];
        min_values[tid] = C_MAX;

        for (size_t j = 0; j < m; ++j) {
            size_t mid = j * WARP_SIZE + tid;
            if (hash2(seed + mid + 1, s_sz, v) % (7) == 1) {
                // atomicAdd(h[mid], 1);
                // if (h[mid] < min[tid]) {
                //     min[tid] = h[mid];
                // }
                min_values[tid] = min(min_values[tid], h[mid]);
            }
        } 

        // C ans = C_MAX;
        // if (tid == 0) {
        //     for (size_t j = 0; j < WARP_SIZE; ++j) {
        //         if (min[j] < ans) {
        //             ans = min[j];
        //         }
        //     }
        //     count[i] = ans;
        // }
        for (int j = 16; j >= 1; j /= 2) {
            C t = __shfl_down_sync(0xffffffff, min_values[tid], j);
            if (tid < j) {
                min_values[tid] = min(min_values[tid], t);
            }
        }

        count[i] = min_values[0];
    }
}


__global__ void random(T *keys, size_t sz, size_t work_load_per_thread) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = tid * work_load_per_thread;
    size_t e = (tid + 1) * work_load_per_thread;
    e = (e <= sz) ? e : sz;

    for (size_t i = b; i < e; ++i) {
        keys[i] *= 17;
    }
}


__global__ void latency_hiding(T *keys, size_t sz, size_t work_load_per_warp, size_t l) {
	__shared__ T wa[WARP_SIZE];
	__shared__ T wb[WARP_SIZE];

	size_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t tid = threadIdx.x % WARP_SIZE;

    wa[tid] = 0;
    wb[tid] = 0;

    size_t b = wid * work_load_per_warp;
    size_t e = (wid + 1) * work_load_per_warp;
    e = min(e, sz);

    for (size_t i = b; i < e; i += WARP_SIZE) {
        size_t t = i + tid;
        t = min(t, sz - 1);
    	// wa[tid] = keys[t];
    	wb[tid] = keys[t];

    	for (size_t w = 0; w < WARP_SIZE; ++w) {
	    	for (size_t j = 0; j < l; ++j) {
	    		// wb[tid] = wb[tid] * 17 + 3;
	    		
	    		// if (tid % 7 == 1)
	    		// atomicAdd(wb + tid, 1);
	    	}
	    	wa[tid] += wb[tid];
	    }
    }

}

__global__ void test() {
    printf("test\n");
}

int main(int argc, char const *argv[])
{

    T *keys;
    size_t keys_sz = 1024 * 1024;//128 * 1024 *1024;
    size_t insert_loop = 1;
    // size_t keys_sz = 128;
    // size_t insert_loop = 1;

    std::unordered_map<T, C> um;

    checkKernelErrors(cudaMalloc((void **)&keys, sizeof(T) * keys_sz));

    T *seed;
    // T seed_num = std::max(hash_table_num, WARP_SIZE);
    checkKernelErrors(cudaMalloc((void **)&seed, sizeof(T) * s_sz * nseed));
    T *tk = new T[keys_sz];
    T *ts = new T[s_sz * nseed];
    {
        char seed_buf[128];
        std::ifstream ifs("/dev/urandom");
        ifs.read(seed_buf, 128);
        std::seed_seq sq(seed_buf, seed_buf + 128);
        std::default_random_engine gen(sq);
        
        {
            // keys
            
            std::uniform_int_distribution<T> dis(0, std::numeric_limits<T>::max());

            for (int i = 0; i < keys_sz; ++i)
            {
                tk[i] = (dis(gen));
                ++um[tk[i]];
            }

            checkKernelErrors(cudaMemcpy(keys, tk, sizeof(T) * keys_sz, cudaMemcpyHostToDevice));

            // delete [] tk;

        }

        {
            // seed
            std::uniform_int_distribution<T> dis(1, std::numeric_limits<T>::max());

            for (int i = 0; i < s_sz * nseed; ++i)
            {
                ts[i] = (dis(gen));
                // std::cout << ts[i] << ",";
            }
            // std::cout << std::endl;

            checkKernelErrors(cudaMemcpy(seed, ts, sizeof(T) * s_sz * nseed, cudaMemcpyHostToDevice));

            // delete [] ts;
        }
        
       
    }

    {
        // debug
    }

    C *hash_table;
    checkKernelErrors(cudaMalloc((void **)&hash_table, sizeof(C) * n_alignment * m * WARP_SIZE));
    checkKernelErrors(cudaMemset(hash_table, 0, sizeof(C) * n_alignment * m * WARP_SIZE));

    C *count;
    checkKernelErrors(cudaMalloc((void **)&count, sizeof(C) * keys_sz));
    checkKernelErrors(cudaMemset(count, 0, sizeof(C) * keys_sz));


    {

        size_t work_load_per_warp = (keys_sz + (nwarps - 1)) / nwarps;
        std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;

        for (size_t l = 2; l <= 1024; l *= 2)
        {
        	using namespace std::chrono;
	        high_resolution_clock::time_point t1 = high_resolution_clock::now();
	        
	        checkKernelErrors((latency_hiding<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp,
	             l)));

	        // for (size_t l = 1; l < insert_loop; ++l) {
	        //     checkKernelErrors((random<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_thread)));
	        //     checkKernelErrors((insert2<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp, hash_table, hash_table_num, hash_table_sz, seed, s_sz, nullptr)));
	        // }
	        cudaDeviceSynchronize();

	        high_resolution_clock::time_point t2 = high_resolution_clock::now();
	        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	        // size_t total_bytes = sizeof(T) * keys_sz * hash_table_num * insert_loop;
	        size_t insert_number = keys_sz * insert_loop;
	        // std::cout << "total_bytes: " << total_bytes << "-->" << (total_bytes / (1ul<<30)) << std::endl;
	        std::cout << "latency_hiding perf: " << double(insert_number) / time_span.count() / (1ul<<20) << "Mops/s" << std::endl;

        }
        
    }


    // {

    //     size_t work_load_per_warp = (keys_sz + (nwarps - 1)) / nwarps;
    //     std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;

    //     std::cout << "p1" << std::endl;
    //     using namespace std::chrono;
    //     high_resolution_clock::time_point t1 = high_resolution_clock::now();
        
    //     checkKernelErrors((search<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp,
    //          hash_table, n, m, seed, s_sz, count, C_MAX, nullptr)));

    //     // for (size_t l = 1; l < insert_loop; ++l) {
    //     //     checkKernelErrors((random<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_thread)));
    //     //     checkKernelErrors((insert2<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp, hash_table, hash_table_num, hash_table_sz, seed, s_sz, nullptr)));
    //     // }
    //     cudaDeviceSynchronize();

    //     high_resolution_clock::time_point t2 = high_resolution_clock::now();
    //     duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    //     // size_t total_bytes = sizeof(T) * keys_sz * hash_table_num * insert_loop;
    //     size_t insert_number = keys_sz * insert_loop;
    //     // std::cout << "total_bytes: " << total_bytes << "-->" << (total_bytes / (1ul<<30)) << std::endl;
    //     std::cout << "search perf: " << double(insert_number) / time_span.count() / (1ul<<20) << "Mops/s" << std::endl;

    //     {
    //         // check
    //         C *ans = new C[keys_sz];
    //         checkKernelErrors(cudaMemcpy(ans, count, sizeof(C) * keys_sz, cudaMemcpyDeviceToHost));
    //         long diff = 0;
    //         for (size_t i = 0; i < keys_sz; ++i) {
    //             if (ans[i] < um[tk[i]]) {
    //                 std::cout << "err" << std::endl;
    //                 break;
    //             } else {
    //                 diff += (ans[i] - um[tk[i]]);
    //             }
    //         }

    //         std::cout << diff / keys_sz << std::endl;
    //         delete [] ans;
    //     }

    // }

    // {

    //     size_t work_load_per_warp = (keys_sz + (nwarps - 1)) / nwarps;
    //     std::cout << "work_load_per_warp: " << work_load_per_warp << std::endl;

    //     std::cout << "p1" << std::endl;
    //     using namespace std::chrono;
    //     high_resolution_clock::time_point t1 = high_resolution_clock::now();
        
    //     checkKernelErrors((search2<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp,
    //          hash_table, n, m, seed, s_sz, count, C_MAX, nullptr)));

    //     // for (size_t l = 1; l < insert_loop; ++l) {
    //     //     checkKernelErrors((random<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_thread)));
    //     //     checkKernelErrors((insert2<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_warp, hash_table, hash_table_num, hash_table_sz, seed, s_sz, nullptr)));
    //     // }
    //     cudaDeviceSynchronize();

    //     high_resolution_clock::time_point t2 = high_resolution_clock::now();
    //     duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    //     // size_t total_bytes = sizeof(T) * keys_sz * hash_table_num * insert_loop;
    //     size_t insert_number = keys_sz * insert_loop;
    //     // std::cout << "total_bytes: " << total_bytes << "-->" << (total_bytes / (1ul<<30)) << std::endl;
    //     std::cout << "search2 perf: " << double(insert_number) / time_span.count() / (1ul<<20) << "Mops/s" << std::endl;

    //     {
    //         // check
    //         C *ans = new C[keys_sz];
    //         checkKernelErrors(cudaMemcpy(ans, count, sizeof(C) * keys_sz, cudaMemcpyDeviceToHost));
    //         long diff = 0;
    //         for (size_t i = 0; i < keys_sz; ++i) {
    //             if (ans[i] < um[tk[i]]) {
    //                 std::cout << "err" << std::endl;
    //                 break;
    //             } else {
    //                 diff += (ans[i] - um[tk[i]]);
    //             }
    //         }

    //         std::cout << diff / keys_sz << std::endl;
    //         delete [] ans;
    //     }

    // }

    delete [] tk;
    delete [] ts;
    


    // std::cout << "p2" << std::endl;

    // checkKernelErrors((search<<<nblocks, threads_per_block>>>(keys, keys_sz, work_load_per_thread, hash_table, hash_table_num, hash_table_sz, seed, s_sz, count,nullptr)));

    // std::cout << "p3" << std::endl;


    // C *count_h = new C[keys_sz];
    // cudaMemcpy(count_h, count, sizeof(C) * keys_sz, cudaMemcpyDeviceToHost);

    // for (size_t i = 0; i < keys_sz; ++i) {
    //     std::cout << count_h[i] << ", ";
    // }
    // std::cout << std::endl;

    // cudaDeviceReset();

    return 0;
}