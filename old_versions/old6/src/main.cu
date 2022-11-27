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
using Insert_Kernel_Function = Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Insert_Kernel_Function;
using Search_Kernel_Function = Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Search_Kernel_Function;
using Insert_Kernel_Function_Levels = Count_Min_Sketch_GPU_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Insert_Kernel_Function;
using Search_Kernel_Function_Levels = Count_Min_Sketch_GPU_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Search_Kernel_Function;
using Insert_Kernel_Function_Mem_Levels = Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Insert_Kernel_Function;
using Search_Kernel_Function_Mem_Levels = Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Search_Kernel_Function;

int main(int argc, char const *argv[])
{
    // {
    //     size_t insert_keys_size = 1024;
    //     Key_T *keys = cpu_tool<Key_T>.random_freq(nullptr, insert_keys_size,
    //         std::numeric_limits<Key_T>::min(),std::numeric_limits<Key_T>::max(),
    //     0.5, 1, 1, 4, 16);

    //     std::unordered_map<Key_T, size_t> um;
    //     for (size_t i = 0; i < insert_keys_size; ++i) {
    //         // std::cout << keys[i] << std::endl;
    //         um[keys[i]]++;
    //     }

    //     for (auto &u : um) {
    //         std::cout << u.first << " : " << u.second << std::endl;
    //     }
    // }

    // return 0;


    // size_t insert_keys_size = 32 * 1024 * 1024;
    // size_t insert_keys_size = 1024 * 1024;
    // size_t search_keys_size = 1024 * 1024;

    size_t insert_keys_size = 1024 * 1024;
    size_t search_keys_size = 1024 * 1024;
    // size_t insert_keys_size = 134217728;
    // size_t insert_keys_size = 4 * 1024;
    // size_t search_keys_size = 32;
    size_t insert_loop = 1;
    size_t search_loop = 1;

    // size_t n = 8388593;
    size_t n = 65521;
    // size_t n = 97;
    size_t seed_sz = 1;
    size_t m = 1;

    // {
    //     // size_t n = 65521;
    //     // size_t seed_sz = 4;
    //     // size_t m = 1;

    //     Unordered_Map_GPU<Key_T, Count_T> emap;

    //     Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
    //         static_cast<Insert_Kernel_Function_Mem_Levels *>(insert_warp_mem_threadfence_preload<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
    //         static_cast<Search_Kernel_Function_Mem_Levels *>(search_warp_mem_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
    //         n, 1024 * 1024, seed_sz);

    //     Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));

    //     // for (size_t insert_keys_size =  1 * 1024 * 1024; insert_keys_size <= 512 * 1024 * 1024; insert_keys_size *= 2) {
    //     // for (size_t insert_keys_size = 120 * 1024 * 1024; insert_keys_size <= 256 * 1024 * 1024; insert_keys_size += 8 * 1024 * 1024) {
    //     //     std::cout << (insert_keys_size / 1024 / 1024) << std::endl;
    //     //     test.insert_perf_test(insert_keys_size, 1);
    //     //     test.clear();
    //     //     std::cout << "----------------------------------" << std::endl;
    //     // }

    //     // test.search_accuracy_test_large(insert_keys_size, search_keys_size);
    //     test.insert_perf_test(insert_keys_size, 1);
    //     // test.clear();
    //     // test.search_perf_test(insert_keys_size, search_keys_size, 1);
    //     // test.clear();
    //     // test.search_accuracy_test(insert_keys_size, search_keys_size, insert_loop);
    //     // test.search_accuracy_freq_test(insert_keys_size, search_keys_size, 1);
    //     // test.clear(); 

    // }

    {
        // size_t n = 65521;
        // size_t seed_sz = 4;
        // size_t m = 1;

        Unordered_Map_GPU<Key_T, Count_T> emap;

        Count_Min_Sketch_GPU_Mem_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
            static_cast<Insert_Kernel_Function_Mem_Levels *>(insert_warp_mem_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            static_cast<Search_Kernel_Function_Mem_Levels *>(search_warp_mem_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
            n, 1024 * 1024, seed_sz);

        Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));

        // for (size_t insert_keys_size = 1024; insert_keys_size <= 256 * 1024; insert_keys_size += 1024) {
        //     std::cout << insert_keys_size / 1024 << std::endl;
        //     // test.search_accuracy_test(insert_keys_size, search_keys_size, insert_loop);
        //     test.search_accuracy_freq_test(insert_keys_size, search_keys_size, 1);
        //     test.clear();
        // }

        // for (size_t insert_keys_size =  1 * 1024 * 1024; insert_keys_size <= 512 * 1024 * 1024; insert_keys_size *= 2) {
        // for (size_t insert_keys_size = 104 * 1024 * 1024; insert_keys_size <= 192 * 1024 * 1024; insert_keys_size += 4 * 1024 * 1024) {
            // std::cout << (insert_keys_size / 1024 / 1024) << "\t"; //<< std::endl;
            // test.insert_perf_test(insert_keys_size, 1);
            // test.insert_perf_test(insert_keys_size, 1);
            // test.clear();
            // test.search_perf_test(insert_keys_size, search_keys_size, 1);
            // test.clear();
            // std::cout << "----------------------------------" << std::endl;
        // }

        // test.search_accuracy_test_large(insert_keys_size, search_keys_size);
        // test.insert_perf_test(insert_keys_size, 1);
        // test.clear();
        // test.search_perf_test(insert_keys_size, search_keys_size, 1);
        // test.clear();
        // test.search_accuracy_test(insert_keys_size, search_keys_size, insert_loop);
        test.search_accuracy_freq_test(insert_keys_size, search_keys_size, 1);
        // test.clear(); 


        // for (int i = 0; i < 100; ++i) {
        //     std::cout << "i: " << i << std::endl;
        //     // test.search_accuracy_test(insert_keys_size, search_keys_size, insert_loop);
        //     test.search_accuracy_freq_test(insert_keys_size, search_keys_size, 1);
        //     test.clear(); 
        // }
    }

    // {
    //     // size_t n = 65521;
    //     // size_t seed_sz = 4;
    //     // size_t m = 1;

    //     Unordered_Map_GPU<Key_T, Count_T> emap;

    //     Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
    //         static_cast<Insert_Kernel_Function *>(insert_warp_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
    //         static_cast<Search_Kernel_Function *>(search_warp_min<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
    //         n, m, seed_sz);

    //     Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));

    //     test.insert_perf_test(insert_keys_size, 1);
    //     test.clear();
    //     // test.clear();
    //     // test.search_perf_test(insert_keys_size, search_keys_size, 1);
    //     // test.clear();
    //     // test.accuracy_test_freq(insert_keys_size, search_keys_size, insert_loop, search_loop);
    //     // // test.search_accuracy_freq_test(insert_keys_size, search_keys_size, 1);
    //     // test.clear(); 

    // }

    // {
    //     // size_t n = 65521;
    //     // size_t seed_sz = 4;
    //     // size_t m = 1;

    //     Unordered_Map_GPU<Key_T, Count_T> emap;

    //     Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
    //         static_cast<Insert_Kernel_Function *>(insert_warp_low_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
    //         static_cast<Search_Kernel_Function *>(search_warp_low_min_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
    //         n, m, seed_sz);

    //     Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));

    //     // test.insert_perf_test(insert_keys_size, 1);
    //     // test.clear();
    //     // test.search_perf_test(insert_keys_size, search_keys_size, 1);
    //     // test.clear();
    //     // test.accuracy_test_freq(insert_keys_size, search_keys_size, insert_loop, search_loop);
    //     // test.search_accuracy_freq_test(insert_keys_size, search_keys_size, 1);

    //     for (int i = 0; i < 100; ++i) {
    //         std::cout << "i: " << i << std::endl;
    //         // test.search_accuracy_test(insert_keys_size, search_keys_size, insert_loop);
    //         test.search_accuracy_freq_test(insert_keys_size, search_keys_size, 1);

    //         test.clear(); 
    //     }

        

    // }


    // {


    //     Unordered_Map_GPU<Key_T, Count_T> emap;

    //     Count_Min_Sketch_GPU_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
    //         static_cast<Insert_Kernel_Function_Levels *>(insert_warp_two_levels_threadfence_byte<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
    //         static_cast<Search_Kernel_Function_Levels *>(search_warp_two_levels_threadfence_byte<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
    //         n, m, n, m, seed_sz);

    //     Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));

    //     // test.insert_perf_test(insert_keys_size, 1);
    //     // test.clear();
    //     // test.search_perf_test(insert_keys_size, search_keys_size, 1);
    //     // test.clear();
    //     test.accuracy_test_freq(insert_keys_size, search_keys_size, insert_loop, search_loop);
    //     // test.search_accuracy_freq_test(insert_keys_size, search_keys_size, 1);
    //     // test.clear(); 

    // }


    // {


    //     Unordered_Map_GPU<Key_T, Count_T> emap;

    //     Count_Min_Sketch_GPU_Levels<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
    //         static_cast<Insert_Kernel_Function_Levels *>(insert_warp_two_levels_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
    //         static_cast<Search_Kernel_Function_Levels *>(search_warp_two_levels_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
    //         n, m, n, m, seed_sz);

    //     Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));

    //     // test.insert_perf_test(insert_keys_size, 1);
    //     // test.clear();
    //     // test.search_perf_test(insert_keys_size, search_keys_size, 1);
    //     // test.clear();
    //     test.accuracy_test_freq(insert_keys_size, search_keys_size, insert_loop, search_loop);
    //     // test.search_accuracy_freq_test(insert_keys_size, search_keys_size, 1);
    //     // test.clear(); 

    // }
    return 0;
}

// int main4(int argc, char const *argv[])
// {
//     size_t insert_keys_size = 1024 * 1024;
//     size_t search_keys_size = 1024 * 1024;

//     {
//         size_t n = 65521;
//         size_t seed_sz = 4;
//         size_t m = 1;

//         Count_Min_Sketch_GPU_Host_Sim<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map_host(
//                 static_cast<Insert_Kernel_Function *>(insert_warp_min_host<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                 static_cast<Search_Kernel_Function *>(search_warp_min_host<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                 n, m, seed_sz);

//         Unordered_Map<Key_T, Count_T> emap;

//         Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map_device(
//                 static_cast<Insert_Kernel_Function *>(insert_warp_min_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                 static_cast<Search_Kernel_Function *>(search_warp_min<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                 n, m, seed_sz);

//         Device_vs_Host_Test<Key_T, Count_T, Hashed_T, Seed_T> test(cpu_tool<Key_T>, map_host, emap, map_device);
//         test.insert_test<decltype(map_host), decltype(map_device)>(insert_keys_size, 1);

//     }

//     return 0;
// }

// int main3(int argc, char const *argv[])
// {
    
//     size_t insert_keys_size = 1024 * 1024;
//     size_t search_keys_size = 1024 * 1024;
//     size_t seed_sz = 4;

//     size_t insert_loop = 64;
//     size_t search_loop = 1;

//     std::vector<size_t> vn {3999971};//65521, 
//     std::vector<size_t> vm {1};

//     while (insert_loop <= 1024) {
//         std::cout << insert_loop << " : ---------------------------------------" << std::endl;
//         for (auto n : vn) {
//             for (auto m : vm) {
//                 {
//                     std::cout << "insert_warp_min: " << std::endl;
//                     std::cout << "insert_keys_size: " << insert_keys_size << std::endl;
//                     std::cout << "search_keys_size: " << search_keys_size << std::endl;

//                     size_t total_slots = n * m * WARP_SIZE;

//                     std::cout << "gpu memory: " << total_slots << "Bytes" << std::endl;

//                     Unordered_Map_GPU<Key_T, Count_T> emap;

//                     Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map4(
//                         static_cast<Insert_Kernel_Function *>(insert_warp_min_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                         static_cast<Search_Kernel_Function *>(search_warp_min<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                         n, m, seed_sz);

//                     Test<Key_T, Count_T, Hashed_T> test4(gpu_tool<Key_T>, (map4), (emap));

//                     // test4.insert_perf_test(insert_keys_size, 1);
//                     // test4.clear();
//                     // test4.search_perf_test(insert_keys_size, search_keys_size, 1);
//                     // test4.clear();
//                     test4.accuracy_test(insert_keys_size, search_keys_size, insert_loop, search_loop);
//                     test4.clear();
//                 }
                
//                 {
//                     std::cout << "insert_warp: " << std::endl;
//                     std::cout << "insert_keys_size: " << insert_keys_size << std::endl;
//                     std::cout << "search_keys_size: " << search_keys_size << std::endl;
//                     size_t total_slots = n * m * WARP_SIZE;

//                     std::cout << "gpu memory: " << total_slots << "Bytes" << std::endl;

//                     Unordered_Map_GPU<Key_T, Count_T> emap;

//                     Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map4(
//                         static_cast<Insert_Kernel_Function *>(insert_warp_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                         static_cast<Search_Kernel_Function *>(search_warp_min<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                         n, m, seed_sz);

//                     Test<Key_T, Count_T, Hashed_T> test4(gpu_tool<Key_T>, (map4), (emap));
//                     // test4.insert_perf_test(insert_keys_size, 1);
//                     // test4.clear();
//                     // test4.search_perf_test(insert_keys_size, search_keys_size, 1);
//                     // test4.clear();
//                     test4.accuracy_test(insert_keys_size, search_keys_size, insert_loop, search_loop);
//                     test4.clear();
//                 }
//             }
//         }

//         insert_loop *= 2;
//         // insert_keys_size *= 2;
//         // search_keys_size *= 2;
//     }

//     return 0;
// }


int main2(int argc, char const *argv[])
{
    // cudaDeviceReset();
    // size_t insert_keys_size = 32;
    // size_t search_keys_size = 32;
    // size_t insert_keys_size = 1024 * 1024;
    // size_t search_keys_size = 1024 * 1024;
    // size_t search_keys_size = 1024;

    // std::cout << "search_accuracy_test: " << std::endl;
    // for (size_t search_keys_size = 1 * 1024 * 1024; search_keys_size <= 256 * 1024 * 1024; search_keys_size *= 2)
    {
        // std::cout << "search_keys_size: " << search_keys_size << std::endl;
        // size_t n = 65521;
        // // size_t n = 3999971;
        // size_t seed_sz = 4;
        // size_t m = 1;
        // std::cout << "n: " << n << ", m: " << m << ", seed_sz: " << seed_sz << std::endl;

        // {
        //     std::cout << "insert_warp_min_host: " << std::endl;
        //     size_t total_slots = n * m * WARP_SIZE;

        //     std::cout << "cpu memory: " << total_slots << "Bytes" << std::endl;

        //     Unordered_Map<Key_T, Count_T> emap;

        //     Count_Min_Sketch_GPU_Host_Sim<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map4(
        //         static_cast<Insert_Kernel_Function *>(insert_warp_min_host<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
        //         static_cast<Search_Kernel_Function *>(search_warp_min_host<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
        //         n, m, seed_sz);

        //     Test<Key_T, Count_T, Hashed_T> test4(cpu_tool<Key_T>, (map4), (emap));

        //     // test4.insert_perf_test(insert_keys_size, 1);
        //     // test4.clear();
        //     // test4.search_perf_test(insert_keys_size, search_keys_size, 1);
        //     // test4.clear();
        //     test4.search_accuracy_test(insert_keys_size, search_keys_size, 1);

            

        // }

        // {
        //     std::cout << "insert_warp_host: " << std::endl;
        //     size_t total_slots = n * m * WARP_SIZE;

        //     std::cout << "cpu memory: " << total_slots << "Bytes" << std::endl;

        //     Unordered_Map<Key_T, Count_T> emap;

        //     Count_Min_Sketch_GPU_Host_Sim<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map4(
        //         static_cast<Insert_Kernel_Function *>(insert_warp_host<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
        //         static_cast<Search_Kernel_Function *>(search_warp_min_host<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
        //         n, m, seed_sz);

        //     Test<Key_T, Count_T, Hashed_T> test4(cpu_tool<Key_T>, (map4), (emap));

        //     // test4.insert_perf_test(insert_keys_size, 1);
        //     // test4.clear();
        //     // test4.search_perf_test(insert_keys_size, search_keys_size, 1);
        //     // test4.clear();
        //     test4.search_accuracy_test(insert_keys_size, search_keys_size, 1);
        // }

       

        // // return 0;

        // {
        //     std::cout << "insert_warp_min: " << std::endl;
        //     size_t total_slots = n * m * WARP_SIZE;

        //     std::cout << "gpu memory: " << total_slots << "Bytes" << std::endl;

        //     Unordered_Map_GPU<Key_T, Count_T> emap;

        //     Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map4(
        //         static_cast<Insert_Kernel_Function *>(insert_warp_min_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
        //         static_cast<Search_Kernel_Function *>(search_warp_min<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
        //         n, m, seed_sz);

        //     Test<Key_T, Count_T, Hashed_T> test4(gpu_tool<Key_T>, (map4), (emap));

        //     // test4.insert_perf_test(insert_keys_size, 1);
        //     // test4.clear();
        //     // test4.search_perf_test(insert_keys_size, search_keys_size, 1);
        //     // test4.clear();
        //     test4.search_accuracy_test(insert_keys_size, search_keys_size, 1);

            

        // }


        // {
        //     std::cout << "insert_warp: " << std::endl;
        //     size_t total_slots = n * m * WARP_SIZE;

        //     std::cout << "gpu memory: " << total_slots << "Bytes" << std::endl;

        //     Unordered_Map_GPU<Key_T, Count_T> emap;

        //     Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map4(
        //         static_cast<Insert_Kernel_Function *>(insert_warp_threadfence<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
        //         static_cast<Search_Kernel_Function *>(search_warp_min<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
        //         n, m, seed_sz);

        //     Test<Key_T, Count_T, Hashed_T> test4(gpu_tool<Key_T>, (map4), (emap));
        //     // test4.insert_perf_test(insert_keys_size, 1);
        //     // test4.clear();
        //     // test4.search_perf_test(insert_keys_size, search_keys_size, 1);
        //     // test4.clear();
        //     test4.search_accuracy_test(insert_keys_size, search_keys_size, 1);

        // }

        // {
        //     std::cout << "insert_cpu: " << std::endl;
        //     size_t cn = 4 * 65521 * 32 / 3;
        //     size_t cnum = 3;
        //     size_t ss = seed_sz;
        //     size_t sn = 128;

        //     size_t total_slots = cn * cnum;
        //     std::cout << "cpu memory: " << total_slots << "Bytes" << std::endl;

        //     Count_Min_Sketch_CPU<Key_T, Count_T> map(cn, cnum, ss, sn);
        //     Unordered_Map<Key_T, Count_T> emap;
        //     Test<unsigned int, unsigned int> test(cpu_tool<Key_T>, (map), (emap));

        //     // test.insert_perf_test(insert_keys_size, 1);
        //     // test.clear();
        //     // test.search_perf_test(insert_keys_size, search_keys_size, 1);
        //     // test.clear();
        //     test.search_accuracy_test(insert_keys_size, search_keys_size, 1);

        //     // test.insert_perf_test(insert_keys_size, 3);
        //     // test.clear();
        //     // test.search_perf_test(insert_keys_size, search_keys_size);
        //     // test.clear();
        //     // test.search_accuracy_test(insert_keys_size, search_keys_size);
        //     // map.print(debug_log);
        // }
    
    }
    return 0;
}


