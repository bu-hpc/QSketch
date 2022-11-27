#include <lib.h>

const dim3 default_grid_dim(65536);
// const dim3 default_grid_dim(1);
const dim3 default_block_dim(32);

constexpr size_t default_hash_table_sz = 65521;
constexpr size_t default_hash_table_num = 5;
constexpr size_t default_seed_sz = 2;
constexpr size_t defualt_seed_num = default_hash_table_num;


constexpr size_t default_hash_n = 65521;
constexpr size_t default_hash_m = 4;
constexpr size_t default_hash_nc = 1024;
constexpr size_t default_hash_nr = 256;


// constexpr size_t host_buf_sz = 1024 * 1024;

std::ofstream debug_log("log.txt");

// template <typename T>
// CPU_Tools<T> cpu_tool;

// template <typename T>
// GPU_Tools<T> gpu_tool;

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

using Key_T = unsigned int; 
using Count_T = unsigned int;
using Hashed_T = unsigned int;
using Seed_T = unsigned int;

using Hash_Function = hash_mul_add<Key_T, Hashed_T, Seed_T>;
// using Hash_Function = hash_xor<Key_T, Hashed_T, Seed_T>;
// using Hash_Function = hash_fast<Key_T, Hashed_T, Seed_T>;
using Insert_Kernel_Function = Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Insert_Kernel_Function;
using Search_Kernel_Function = Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Search_Kernel_Function;
using Insert_Kernel_Function_Levels = Count_Min_Sketch_GPU_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Insert_Kernel_Function;
using Search_Kernel_Function_Levels = Count_Min_Sketch_GPU_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Search_Kernel_Function;
using Insert_Kernel_Function_Mem_Levels = Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Insert_Kernel_Function;
using Search_Kernel_Function_Mem_Levels = Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Search_Kernel_Function;
using Insert_Kernel_Function_Mem_Levels_Pre_Cal = Count_Min_Sketch_GPU_Mem_Levels_Pre_Cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Insert_Kernel_Function;
using Search_Kernel_Function_Mem_Levels_Pre_Cal = Count_Min_Sketch_GPU_Mem_Levels_Pre_Cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Search_Kernel_Function;
using Insert_Kernel_Function_Mem_Levels_Host_Pre_Cal = Count_Min_Sketch_GPU_Mem_Levels_Host_Pre_Cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Insert_Kernel_Function;
using Search_Kernel_Function_Mem_Levels_Host_Pre_Cal = Count_Min_Sketch_GPU_Mem_Levels_Host_Pre_Cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Search_Kernel_Function;
using Insert_Kernel_Function_Mem_Levels_Host_Pre_Cal_Sub_Warp = Count_Min_Sketch_GPU_Mem_Levels_Host_Pre_Cal_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Insert_Kernel_Function;
using Search_Kernel_Function_Mem_Levels_Host_Pre_Cal_Sub_Warp = Count_Min_Sketch_GPU_Mem_Levels_Host_Pre_Cal_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Search_Kernel_Function;
using Insert_Kernel_Function_Mem_Levels_Pre_Load_Host_Pre_Cal_Sub_Warp = Count_Min_Sketch_GPU_Mem_Levels_Pre_Load_Host_Pre_Cal_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Insert_Kernel_Function;
using Search_Kernel_Function_Mem_Levels_Pre_Load_Host_Pre_Cal_Sub_Warp = Count_Min_Sketch_GPU_Mem_Levels_Pre_Load_Host_Pre_Cal_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Search_Kernel_Function;


int warm();
int run_example_3();
int run_example_4();
int run_example_5();
int run_example_bus_width();
int run_preload();
int run_preload_pre_cal();
int run_preload_host_pre_cal();
int run_without_preload_host_pre_cal_sub_warp();
int run_preload_host_pre_cal_sub_warp();

int run_without_preload_host_pre_cal_sub_warp_loop();

int random_nums_gpu();
int random_nums_cpu();

int debug();


// size_t insert_keys_sz = 1024;
// size_t search_keys_sz = 1024;


// small
size_t insert_keys_sz = 1024 * 1024;
size_t search_keys_sz = 1024 * 1024;
// //size_t n = 8191;
size_t n = 65521;
size_t n_sub_warp = 262139;

// // medium
// size_t insert_keys_sz = 32 * 1024 * 1024;
// size_t search_keys_sz = 32 * 1024 * 1024;
// // size_t n = 131071;
// size_t n = 2097143;
// size_t n_sub_warp = 8388593;

// // large
// size_t insert_keys_sz = 128 * 1024 * 1024;
// size_t search_keys_sz = 128 * 1024 * 1024;
// //size_t n = 1048573; //  < 1024 * 1024
// size_t n = 8388593;
// size_t n_sub_warp = 33554393;

// size_t insert_keys_sz = 134217728;
// size_t insert_keys_sz = 1024;
// size_t search_keys_sz = 32;



size_t insert_loop = 1;
size_t search_loop = 1;

// size_t n = 8388593;
// size_t n = 7456549;
// size_t n = 257731;
// size_t n = 1048573; //  < 1024 * 1024
// size_t n = 65521;
// size_t n = 97;
size_t seed_sz = 1;
size_t m = 1;

int main(int argc, char const *argv[])
{

    // size_t fifo_sz;
    // cudaDeviceGetLimit(&fifo_sz,cudaLimitPrintfFifoSize);
    // // std::cout << fifo_sz << std::endl;
    // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, fifo_sz * 8);

    warm();

    // run_example_bus_width();
    // run_preload();

    // run_preload_pre_cal();

    // run_example_5();
    // run_preload_host_pre_cal();
    run_without_preload_host_pre_cal_sub_warp();
    // run_preload_host_pre_cal_sub_warp();
    // run_without_preload_host_pre_cal_sub_warp_loop();
    // random_nums_gpu();

    // debug();

    // insert_warp_mem_threadfence_without_preload_pre_cal_sub_warp(nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 
    //     Hash_Function(), nullptr, 0, nullptr, 0, nullptr);
    return 0;
}

int warm() {
    {
        std::cout << "warm" << std::endl;
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        // size_t insert_keys_sz = 1024 * 1024;
        // size_t search_keys_sz = 1024 * 1024;
                

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels *>(insert_warp_mem_threadfence_preload<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels *>(search_warp_mem_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));
        test.insert_perf_test(insert_keys_sz, 1, std::cerr);
        test.clear();
        test.search_perf_test(insert_keys_sz, search_keys_sz, 1, std::cerr);
        test.clear();
        // test.search_accuracy_test(insert_keys_sz, search_keys_sz, insert_loop);
        // test.clear(); 
        // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear(); 

    }
    std::cout << "----------------------------------" << std::endl;
    return 0;
}




int run_example_3()
{
    {
        // std::cout << "example3_1" << std::endl;
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels *>(insert_warp_mem_threadfence_example3_1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels *>(search_warp_mem_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));

        // for (int i = 0; i < 10000; ++i) 
        {
            // std::cout << "test: " << i << std::endl;
            test.insert_perf_test(insert_keys_sz, 1);
            test.clear();
            // test.insert_perf_freq_test(insert_keys_sz, 1);
            // test.clear();
            // test.search_perf_test(insert_keys_sz, search_keys_sz, 1);
            // test.clear();
            // test.search_accuracy_test(insert_keys_sz, search_keys_sz, insert_loop);
            // test.clear();
            // auto err = test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
            // test.clear();
            // std::cout << "----------------" << i << "------------------" << std::endl;
            // if (err < 0) {
            //     std::cout << "err: " << err << std::endl;
            //     break;
            // }
        }
    }



    {
        std::cout << "example3_1" << std::endl;
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels *>(insert_warp_mem_threadfence_example3_1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels *>(search_warp_mem_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));

        // for (int i = 0; i < 10000; ++i) 
        {
            // std::cout << "test: " << i << std::endl;
            test.insert_perf_test(insert_keys_sz, 1);
            test.clear();
            // test.insert_perf_freq_test(insert_keys_sz, 1);
            // test.clear();
            // test.search_perf_test(insert_keys_sz, search_keys_sz, 1);
            // test.clear();
            // test.search_accuracy_test(insert_keys_sz, search_keys_sz, insert_loop);
            // test.clear();
            // auto err = test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
            // test.clear();
            // std::cout << "----------------" << i << "------------------" << std::endl;
            // if (err < 0) {
            //     std::cout << "err: " << err << std::endl;
            //     break;
            // }
        }
    }
    std::cout << "----------------" << "------------------" << std::endl;

    {
        std::cout << "example3_2" << std::endl;
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels *>(insert_warp_mem_threadfence_example3_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels *>(search_warp_mem_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));

        // for (int i = 0; i < 10000; ++i) 
        {
            // std::cout << "test: " << i << std::endl;
            test.insert_perf_test(insert_keys_sz, 1);
            test.clear();
            // test.insert_perf_freq_test(insert_keys_sz, 1);
            // test.clear();
            // test.search_perf_test(insert_keys_sz, search_keys_sz, 1);
            // test.clear();
            // test.search_accuracy_test(insert_keys_sz, search_keys_sz, insert_loop);
            // test.clear();
            // auto err = test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
            // test.clear();
            // std::cout << "----------------" << i << "------------------" << std::endl;
            // if (err < 0) {
            //     std::cout << "err: " << err << std::endl;
            //     break;
            // }
        }
    }
    std::cout << "----------------" << "------------------" << std::endl;

    {
        std::cout << "example3_3" << std::endl;
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels *>(insert_warp_mem_threadfence_example3_3<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels *>(search_warp_mem_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));

        // for (int i = 0; i < 10000; ++i) 
        {
            // std::cout << "test: " << i << std::endl;
            test.insert_perf_test(insert_keys_sz, 1);
            test.clear();
            // test.insert_perf_freq_test(insert_keys_sz, 1);
            // test.clear();
            // test.search_perf_test(insert_keys_sz, search_keys_sz, 1);
            // test.clear();
            // test.search_accuracy_test(insert_keys_sz, search_keys_sz, insert_loop);
            // test.clear();
            // auto err = test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
            // test.clear();
            // std::cout << "----------------" << i << "------------------" << std::endl;
            // if (err < 0) {
            //     std::cout << "err: " << err << std::endl;
            //     break;
            // }
        }
    }
    std::cout << "----------------" << "------------------" << std::endl;
    return 0;
}
int run_example_5()
{
    {
        std::cout << "run_example_5_1" << std::endl;
        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels *>(insert_warp_mem_threadfence_example5_1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels *>(search_warp_mem_threadfence_example5_1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));

        // for (int i = 0; i < 10000; ++i) 
        {
            test.insert_perf_test(insert_keys_sz, 1);
            test.clear();
            test.search_perf_test(insert_keys_sz, search_keys_sz, 1);
            test.clear();
            // test.search_accuracy_test(insert_keys_sz, search_keys_sz, insert_loop);
            // test.clear();
        }
    }
    std::cout << "----------------------------------" << std::endl;
    
    {
        std::cout << "run_example_5_2" << std::endl;
        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels *>(insert_warp_mem_threadfence_example5_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels *>(search_warp_mem_threadfence_example5_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));

        // for (int i = 0; i < 10000; ++i) 
        {
            test.insert_perf_test(insert_keys_sz, 1);
            test.clear();
            test.search_perf_test(insert_keys_sz, search_keys_sz, 1);
            test.clear();
            // test.search_accuracy_test(insert_keys_sz, search_keys_sz, insert_loop);
            // test.clear();
        }
    }
    std::cout << "----------------------------------" << std::endl;

    return 0;
}


int run_example_bus_width()
{


    {
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels *>(insert_warp_mem_threadfence_preload<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels *>(search_warp_mem_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));

        // for (size_t insert_keys_sz =  1 * 1024 * 1024; insert_keys_sz <= 512 * 1024 * 1024; insert_keys_sz *= 2) {
        // for (size_t insert_keys_sz = 120 * 1024 * 1024; insert_keys_sz <= 256 * 1024 * 1024; insert_keys_sz += 8 * 1024 * 1024) {
        //     std::cout << (insert_keys_sz / 1024 / 1024) << std::endl;
        //     test.insert_perf_test(insert_keys_sz, 1);
        //     test.clear();
        //     std::cout << "----------------------------------" << std::endl;
        // }

        // test.search_accuracy_test_large(insert_keys_sz, search_keys_sz);
        test.insert_perf_test(insert_keys_sz, 1);
        test.clear();
        // test.search_perf_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear();
        // // test.search_accuracy_test(insert_keys_sz, search_keys_sz, insert_loop);
        // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear(); 

    }
    std::cout << "----------------------------------" << std::endl;

    {
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels *>(insert_warp_mem_threadfence_preload<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels *>(search_warp_mem_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));

        // for (size_t insert_keys_sz =  1 * 1024 * 1024; insert_keys_sz <= 512 * 1024 * 1024; insert_keys_sz *= 2) {
        // for (size_t insert_keys_sz = 120 * 1024 * 1024; insert_keys_sz <= 256 * 1024 * 1024; insert_keys_sz += 8 * 1024 * 1024) {
        //     std::cout << (insert_keys_sz / 1024 / 1024) << std::endl;
        //     test.insert_perf_test(insert_keys_sz, 1);
        //     test.clear();
        //     std::cout << "----------------------------------" << std::endl;
        // }

        // test.search_accuracy_test_large(insert_keys_sz, search_keys_sz);
        test.insert_perf_test(insert_keys_sz, 1);
        test.clear();
        test.insert_perf_freq_test(insert_keys_sz, 1);
        test.clear();
        // test.search_perf_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear();
        // // test.search_accuracy_test(insert_keys_sz, search_keys_sz, insert_loop);
        // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear(); 

    }
    std::cout << "----------------------------------" << std::endl;
    {
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels *>(insert_warp_mem_threadfence_without_preload<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels *>(search_warp_mem_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));

        // for (size_t insert_keys_sz =  1 * 1024 * 1024; insert_keys_sz <= 512 * 1024 * 1024; insert_keys_sz *= 2) {
        // for (size_t insert_keys_sz = 120 * 1024 * 1024; insert_keys_sz <= 256 * 1024 * 1024; insert_keys_sz += 8 * 1024 * 1024) {
        //     std::cout << (insert_keys_sz / 1024 / 1024) << std::endl;
        //     test.insert_perf_test(insert_keys_sz, 1);
        //     test.clear();
        //     std::cout << "----------------------------------" << std::endl;
        // }

        test.insert_perf_test(insert_keys_sz, 1);
        test.clear();
        test.insert_perf_freq_test(insert_keys_sz, 1);
        test.clear();

        // test.search_accuracy_test_large(insert_keys_sz, search_keys_sz);
        // test.insert_perf_test(insert_keys_sz, 1);
        // test.clear();
        // test.insert_perf_freq_test(insert_keys_sz, 1);
        // test.clear();
        // test.search_perf_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear();
        // // test.search_accuracy_test(insert_keys_sz, search_keys_sz, insert_loop);
        // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear(); 

    }

    return 0;
}


int run_preload()
{

    // {
    //     std::cout << "old_version" << std::endl;
    //     // size_t n = 65521;
    //     // size_t seed_sz = 4;
    //     // size_t m = 1;

    //     Unordered_Map_GPU<Key_T, Count_T> emap;

    //     Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
    //         static_cast<Insert_Kernel_Function_Mem_Levels *>(insert_warp_mem_threadfence_preload<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
    //         static_cast<Search_Kernel_Function_Mem_Levels *>(search_warp_mem_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
    //         n, 1024 * 1024, seed_sz);

    //     Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));
    //     test.insert_perf_test(insert_keys_sz, 1);
    //     test.clear();
    //     test.search_perf_test(insert_keys_sz, search_keys_sz, 1);
    //     test.clear();
    //     // test.search_accuracy_test(insert_keys_sz, search_keys_sz, insert_loop);
    //     // test.clear(); 
    //     // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
    //     // test.clear(); 

    //     // test.search_accuracy_test(insert_keys_sz, search_keys_sz, insert_loop);
    //     // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
    //     // test.clear(); 

    // }
    // std::cout << "----------------------------------" << std::endl;

    {
        std::cout << "without_preload" << std::endl;
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels *>(insert_warp_mem_threadfence_without_preload<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels *>(search_warp_mem_threadfence_without_preload<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));

        // test.insert_perf_test(insert_keys_sz, 1);
        // test.clear();
        // test.search_perf_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear();
        // test.search_accuracy_test(insert_keys_sz, search_keys_sz, insert_loop);
        // test.clear(); 
        // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear();

        test.insert_perf_freq_test(insert_keys_sz, 1);
        test.clear();
        test.search_perf_freq_test(insert_keys_sz, search_keys_sz, 1);
        test.clear();



        // for (size_t insert_keys_sz =  1 * 1024 * 1024; insert_keys_sz <= 512 * 1024 * 1024; insert_keys_sz *= 2) {
        // for (size_t insert_keys_sz = 120 * 1024 * 1024; insert_keys_sz <= 256 * 1024 * 1024; insert_keys_sz += 8 * 1024 * 1024) {
        //     std::cout << (insert_keys_sz / 1024 / 1024) << std::endl;
        //     test.insert_perf_test(insert_keys_sz, 1);
        //     test.clear();
        //     std::cout << "----------------------------------" << std::endl;
        // }

        // test.search_accuracy_test_large(insert_keys_sz, search_keys_sz);
        // test.insert_perf_test(insert_keys_sz, 1);
        // test.clear();
        // test.insert_perf_freq_test(insert_keys_sz, 1);
        // test.clear();
        // test.search_perf_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear();
        // // test.search_accuracy_test(insert_keys_sz, search_keys_sz, insert_loop);
        // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear(); 

    }
    std::cout << "----------------------------------" << std::endl;
    // return 0;
    {
        std::cout << "preload" << std::endl;
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels *>(insert_warp_mem_threadfence_preload<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels *>(search_warp_mem_threadfence_preload<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));
        // test.insert_perf_test(insert_keys_sz, 1, std::cout);
        // test.clear();
        // test.search_perf_test(insert_keys_sz, search_keys_sz, 1, std::cout);
        // test.clear();
        // test.search_accuracy_test(insert_keys_sz, search_keys_sz, insert_loop);
        // test.clear(); 
        // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear();
        test.insert_perf_freq_test(insert_keys_sz, 1);
        test.clear();
        test.search_perf_freq_test(insert_keys_sz, search_keys_sz, 1);
        test.clear();

        // for (size_t insert_keys_sz =  1 * 1024 * 1024; insert_keys_sz <= 512 * 1024 * 1024; insert_keys_sz *= 2) {
        // for (size_t insert_keys_sz = 120 * 1024 * 1024; insert_keys_sz <= 256 * 1024 * 1024; insert_keys_sz += 8 * 1024 * 1024) {
        //     std::cout << (insert_keys_sz / 1024 / 1024) << std::endl;
        //     test.insert_perf_test(insert_keys_sz, 1);
        //     test.clear();
        //     std::cout << "----------------------------------" << std::endl;
        // }

        // test.insert_perf_test(insert_keys_sz, 1);
        // test.clear();
        // test.insert_perf_freq_test(insert_keys_sz, 1);
        // test.clear();

        // test.search_accuracy_test_large(insert_keys_sz, search_keys_sz);
        // test.insert_perf_test(insert_keys_sz, 1);
        // test.clear();
        // test.insert_perf_freq_test(insert_keys_sz, 1);
        // test.clear();
        // test.search_perf_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear();
        // // test.search_accuracy_test(insert_keys_sz, search_keys_sz, insert_loop);
        // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear(); 

    }
    std::cout << "----------------------------------" << std::endl;
    return 0;
}

int run_preload_pre_cal()
{
    
    {
        std::cout << "without_preload pre_cal" << std::endl;
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels_Pre_Cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels_Pre_Cal *>(insert_warp_mem_threadfence_without_preload_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels_Pre_Cal *>(search_warp_mem_threadfence_without_preload_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));
        // test.insert_perf_test(insert_keys_sz, 1, std::cerr);
        // test.clear();
        // test.search_perf_test(insert_keys_sz, search_keys_sz, 1, std::cerr);
        // test.clear();
        // test.search_accuracy_test_debug(insert_keys_sz, search_keys_sz, insert_loop);
        // test.clear(); 
        // test.search_accuracy_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear(); 
        // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear();
        test.insert_perf_freq_test(insert_keys_sz, 1);
        test.clear();
        test.search_perf_freq_test(insert_keys_sz, search_keys_sz, 1);
        test.clear();

    }
    std::cout << "----------------------------------" << std::endl;

    {
        std::cout << "preload pre_cal" << std::endl;
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels_Pre_Cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels_Pre_Cal *>(insert_warp_mem_threadfence_preload_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels_Pre_Cal *>(search_warp_mem_threadfence_preload_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));
        // test.insert_perf_test(insert_keys_sz, 1, std::cerr);
        // test.clear();
        // test.search_perf_test(insert_keys_sz, search_keys_sz, 1, std::cerr);
        // test.clear();
        // test.search_accuracy_test_debug(insert_keys_sz, search_keys_sz, insert_loop);
        // test.clear(); 
        // test.search_accuracy_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear(); 
        // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear(); 
        test.insert_perf_freq_test(insert_keys_sz, 1);
        test.clear();
        test.search_perf_freq_test(insert_keys_sz, search_keys_sz, 1);
        test.clear();

    }
    std::cout << "----------------------------------" << std::endl;


    
    return 0;
}

int run_preload_host_pre_cal()
{
    
    {
        std::cout << "without_preload host_pre_cal" << std::endl;
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels_Host_Pre_Cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels_Host_Pre_Cal *>(insert_warp_mem_threadfence_without_preload_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels_Host_Pre_Cal *>(search_warp_mem_threadfence_without_preload_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));
        // test.insert_perf_test(insert_keys_sz, 1, std::cerr);
        // test.clear();
        // test.search_perf_test(insert_keys_sz, search_keys_sz, 1, std::cerr);
        // test.clear();
        // test.search_accuracy_test_debug(insert_keys_sz, search_keys_sz, insert_loop);
        // test.clear(); 
        // test.search_accuracy_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear(); 
        // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear();
        test.insert_perf_freq_test(insert_keys_sz, 1);
        test.clear();
        test.search_perf_freq_test(insert_keys_sz, search_keys_sz, 1);
        test.clear();

    }
    std::cout << "----------------------------------" << std::endl;

    {
        std::cout << "preload host_pre_cal" << std::endl;
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels_Host_Pre_Cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels_Host_Pre_Cal *>(insert_warp_mem_threadfence_preload_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels_Host_Pre_Cal *>(search_warp_mem_threadfence_preload_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));
        // test.insert_perf_test(insert_keys_sz, 1, std::cerr);
        // test.clear();
        // test.search_perf_test(insert_keys_sz, search_keys_sz, 1, std::cerr);
        // test.clear();
        // test.search_accuracy_test_debug(insert_keys_sz, search_keys_sz, insert_loop);
        // test.clear(); 
        // test.search_accuracy_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear(); 
        // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear(); 
        test.insert_perf_freq_test(insert_keys_sz, 1);
        test.clear();
        test.search_perf_freq_test(insert_keys_sz, search_keys_sz, 1);
        test.clear();

    }
    std::cout << "----------------------------------" << std::endl;


    
    return 0;
}

int run_without_preload_host_pre_cal_sub_warp() {
    {
        std::cout << "without_preload host_pre_cal sub warp" << std::endl;
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels_Host_Pre_Cal_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels_Host_Pre_Cal_Sub_Warp *>(insert_warp_mem_threadfence_without_preload_pre_cal_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels_Host_Pre_Cal_Sub_Warp *>(search_warp_mem_threadfence_without_preload_pre_cal_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n_sub_warp, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));
        test.insert_perf_test(insert_keys_sz, 1, std::cerr);
        test.clear();
        test.search_perf_test(insert_keys_sz, search_keys_sz, 1, std::cerr);
        test.clear();
        // test.search_accuracy_test_debug(insert_keys_sz, search_keys_sz, insert_loop);
        // test.clear(); 
        test.search_accuracy_test(insert_keys_sz, search_keys_sz, 1);
        test.clear(); 
       
        test.insert_perf_freq_test(insert_keys_sz, 1);
        test.clear();
        test.search_perf_freq_test(insert_keys_sz, search_keys_sz, 1);
        test.clear();
        test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        test.clear();

    }
    std::cout << "----------------------------------" << std::endl;

    return 0;
}

int run_preload_host_pre_cal_sub_warp() {
    {
        std::cout << "preload host_pre_cal sub warp" << std::endl;
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels_Pre_Load_Host_Pre_Cal_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels_Pre_Load_Host_Pre_Cal_Sub_Warp *>(insert_warp_mem_threadfence_preload_pre_cal_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels_Pre_Load_Host_Pre_Cal_Sub_Warp *>(search_warp_mem_threadfence_preload_pre_cal_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n_sub_warp, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));
        test.insert_perf_test(insert_keys_sz, 1, std::cerr);
        test.clear();
        test.search_perf_test(insert_keys_sz, search_keys_sz, 1, std::cerr);
        test.clear();
        // test.search_accuracy_test_debug(insert_keys_sz, search_keys_sz, insert_loop);
        // test.clear(); 
        test.search_accuracy_test(insert_keys_sz, search_keys_sz, 1);
        test.clear(); 
       
        test.insert_perf_freq_test(insert_keys_sz, 1);
        test.clear();
        test.search_perf_freq_test(insert_keys_sz, search_keys_sz, 1);
        test.clear();
        test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        test.clear();

    }
    std::cout << "----------------------------------" << std::endl;

    return 0;
}

int run_without_preload_host_pre_cal_sub_warp_loop() {
    {
        std::cout << "run_without_preload_host_pre_cal_sub_warp_loop" << std::endl;
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels_Host_Pre_Cal_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels_Host_Pre_Cal_Sub_Warp *>(insert_warp_mem_threadfence_without_preload_pre_cal_sub_warp_loop<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels_Host_Pre_Cal_Sub_Warp *>(search_warp_mem_threadfence_without_preload_pre_cal_sub_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n_sub_warp, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));
        test.insert_perf_test(insert_keys_sz, 1, std::cerr);
        test.clear();
        // test.search_perf_test(insert_keys_sz, search_keys_sz, 1, std::cerr);
        // test.clear();
        // test.search_accuracy_test_debug(insert_keys_sz, search_keys_sz, insert_loop);
        // test.clear(); 
        // test.search_accuracy_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear(); 
       
        // test.insert_perf_freq_test(insert_keys_sz, 1);
        // test.clear();
        // test.search_perf_freq_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear();
        // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear();

    }
    std::cout << "----------------------------------" << std::endl;

    return 0;
}



int debug() {
    
    const std::string file_name = "data/gpu_random.dat";
    // {
    //     Key_T *keys = gpu_tool<Key_T>.random(nullptr, insert_keys_sz);
    //     unsigned int *buf = new unsigned int[insert_keys_sz];
    //     checkKernelErrors(cudaMemcpy(buf, keys, sizeof(unsigned int) * insert_keys_sz, cudaMemcpyDeviceToHost));
    //     int rt = write_file(file_name, buf, insert_keys_sz * sizeof(Key_T));
    //     delete []buf;
    //     std::cout << rt << std::endl;
    // }

    // return 0;

    // Key_T *keys_host;
    // size_t file_sz;
    // int rt = read_file(file_name, (void **)&keys_host, file_sz);
    // if (rt < 0) {
    //     std::cout << rt << std::endl;
    // }
    // std::cout << "file_sz: " << file_sz << std::endl;

    // Key_T *keys;
    // checkKernelErrors(cudaMalloc(&keys, sizeof(Key_T) * insert_keys_sz));
    // checkKernelErrors(cudaMemcpy(keys, keys_host, sizeof(Key_T) * insert_keys_sz, cudaMemcpyHostToDevice));
    {
        std::cout << "debug" << std::endl;
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels_Pre_Cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels_Pre_Cal *>(insert_warp_mem_threadfence_without_preload_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels_Pre_Cal *>(search_warp_mem_threadfence_without_preload_pre_cal<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));



        // test.search_accuracy_test_file_debug(keys, insert_keys_sz, search_keys_sz);

        test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        test.clear();
        // test.search_accuracy_test_debug(insert_keys_sz, search_keys_sz, 1);
        // test.insert_perf_test(insert_keys_sz, 1, std::cerr);
        // test.clear();
        // test.search_perf_test(insert_keys_sz, search_keys_sz, 1, std::cerr);
        // test.clear();
        // test.search_accuracy_test_debug(insert_keys_sz, search_keys_sz, insert_loop);
        // test.clear(); 
        // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear(); 
        // test.insert_perf_freq_test(insert_keys_sz, 1);
        // test.clear();
        // test.search_perf_freq_test(insert_keys_sz, search_keys_sz, 1);
        // test.clear();

    }

    return 0;
}

int random_nums_gpu() {
    Key_T *keys = nullptr;
    size_t insert_keys_sz = 1024 * 1024;
    keys = gpu_tool<Key_T>.random(keys, insert_keys_sz);
    unsigned int *buf = new unsigned int[insert_keys_sz];
    cudaDeviceSynchronize();
    checkKernelErrors(cudaMemcpy(buf, keys, sizeof(unsigned int) * insert_keys_sz, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < insert_keys_sz; ++i) {
        // std::cout << buf[i] << std::endl;
        std::bitset<32> bs(buf[i]);
        std::cout << bs << "\t:\t" << buf[i] << std::endl;
    }
    return 0;
}

int random_nums_cpu() {
    Key_T *keys = nullptr;
    size_t insert_keys_sz = 1024 * 1024;
    keys = gpu_tool<Key_T>.random(keys, insert_keys_sz);
    unsigned int *buf = new unsigned int[insert_keys_sz];
    cudaDeviceSynchronize();
    checkKernelErrors(cudaMemcpy(buf, keys, sizeof(unsigned int) * insert_keys_sz, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < insert_keys_sz; ++i) {
        // std::cout << buf[i] << std::endl;
        std::bitset<32> bs(buf[i]);
        std::cout << bs << "\t:\t" << buf[i] << std::endl;
    }
    return 0;
}