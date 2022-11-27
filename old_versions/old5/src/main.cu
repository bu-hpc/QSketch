#include <lib.h>

const dim3 default_grid_dim(65536);
const dim3 default_block_dim(32);

constexpr size_t default_hash_table_sz = 65521;
constexpr size_t default_hash_table_num = 5;
constexpr size_t default_seed_sz = 2;
constexpr size_t defualt_seed_num = default_hash_table_num;


constexpr size_t default_hash_n = 65521;
constexpr size_t default_hash_m = 4;

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

int main4(int argc, char const *argv[])
{

    
    size_t hs = 97;
    size_t hn = 4;
    size_t ss = 8;
    size_t sn = 128;

    size_t total_slots = hs * hn;
    std::cout << "cpu : " << total_slots << std::endl;

    Count_Min_Sketch_CPU<Key_T, Count_T> map(hs, hn, ss, sn);
    Unordered_Map<Key_T, Count_T> emap;
    Test<unsigned int, unsigned int> test(cpu_tool<Key_T>, (map), (emap));
    
    // size_t insert_keys_size = 32;
    // size_t search_keys_size = 32;

    for (size_t insert_keys_size = 32; insert_keys_size <= 2048; insert_keys_size *= 2) {
        std::cout << "insert_keys_size: " << insert_keys_size << std::endl;
        size_t search_keys_size = insert_keys_size / 4;
        test.search_accuracy_test(insert_keys_size, search_keys_size);
        test.clear();
    }


        
    
    
    return 0;
}

int main(int argc, char const *argv[])
{
    // cudaDeviceReset();
    // size_t insert_keys_size = 32;
    // size_t search_keys_size = 32;
    size_t insert_keys_size = 1024 * 1024;
    size_t search_keys_size = 1024;
    // size_t search_keys_size = 1024;

    // std::cout << "search_accuracy_test: " << std::endl;
    {
        size_t n = 65521;
        size_t seed_sz = 8;
        size_t m = 1;
        // std::cout << "n: " << n << ", m: " << m << ", seed_sz: " << seed_sz << std::endl;

        {
            // std::cout << "insert_warp_min: " << std::endl;
            size_t total_slots = n * m * WARP_SIZE;

            std::cout << "gpu : " << total_slots << std::endl;

            Unordered_Map_GPU<Key_T, Count_T> emap;

            Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map4(
                static_cast<Insert_Kernel_Function *>(insert_warp_min_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
                static_cast<Search_Kernel_Function *>(search_warp_test_min_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
                n, m, seed_sz);

            Test<Key_T, Count_T, Hashed_T> test4(gpu_tool<Key_T>, (map4), (emap));
            test4.search_accuracy_test(insert_keys_size, search_keys_size, 1);
            

        }

        {
            size_t cn = 65521 * 8;
            size_t cnum = 4;
            size_t ss = seed_sz;
            size_t sn = 128;

            size_t total_slots = cn * cnum;
            std::cout << "cpu : " << total_slots << std::endl;

            Count_Min_Sketch_CPU<Key_T, Count_T> map(cn, cnum, ss, sn);
            Unordered_Map<Key_T, Count_T> emap;
            Test<unsigned int, unsigned int> test(cpu_tool<Key_T>, (map), (emap));

            // test.insert_perf_test(insert_keys_size, 3);
            // test.clear();
            // test.search_perf_test(insert_keys_size, search_keys_size);
            // test.clear();
            test.search_accuracy_test(insert_keys_size, search_keys_size);
            // map.print(debug_log);
        }
    
    }
    return 0;
}


// int main3(int argc, char const *argv[])
// {
//     // cudaDeviceReset();
//     size_t insert_keys_size = 512 * 1024 * 1024;
//     size_t search_keys_size = 512 * 1024 * 1024;
//     // size_t insert_keys_size = 32 * 1024;
//     // size_t search_keys_size = 1024;

//     using Key_T = unsigned int; 
//     using Count_T = unsigned int;
//     using Hashed_T = unsigned int;
//     using Seed_T = unsigned int;

//     using Hash_Function = hash_mul_add<Key_T, Hashed_T, Seed_T>;
//     using Insert_Kernel_Function = Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Insert_Kernel_Function;
//     using Search_Kernel_Function = Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Search_Kernel_Function;

//     {
//         // size_t n = 268435399;
//         size_t n = 65521;
//         // size_t n = 3999971;
//         size_t seed_sz = 2;
//         size_t m = 4;

//         for (;;)
//         // for (size_t m = 1; m <= 8; m *= 2) 
//         {

//             std::cout << "n: " << n << ", m: " << m << ", seed_sz: " << seed_sz << std::endl;
//             Unordered_Map_GPU<Key_T, Count_T> emap;
            


//             // Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map3(
//             //     static_cast<Insert_Kernel_Function *>(insert_warp_min<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//             //     static_cast<Search_Kernel_Function *>(search_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//             //     n, m, seed_sz);

//             // Test<Key_T, Count_T, Hashed_T> test3(gpu_tool<Key_T>, (map3), (emap));
//             // test3.search_perf_test(insert_keys_size, search_keys_size);
//             // test3.clear();


//             Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map4(
//                 static_cast<Insert_Kernel_Function *>(insert_warp_min<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                 static_cast<Search_Kernel_Function *>(search_warp_test_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                 n, m, seed_sz);

//             Test<Key_T, Count_T, Hashed_T> test4(gpu_tool<Key_T>, (map4), (emap));
//             test4.search_perf_test(insert_keys_size, search_keys_size);
//             test4.clear();
//         }
//     }

//     return 0;
// }

// int main2(int argc, char const *argv[])
// {
//     // cudaDeviceReset();
//     size_t insert_keys_size = 512 * 1024 * 1024;
//     size_t search_keys_size = 32 * 1024 * 1024;
//     // size_t insert_keys_size = 32 * 1024;
//     // size_t search_keys_size = 1024;

//     using Key_T = unsigned int; 
//     using Count_T = unsigned int;
//     using Hashed_T = unsigned int;
//     using Seed_T = unsigned int;

//     using Hash_Function = hash_mul_add<Key_T, Hashed_T, Seed_T>;
//     using Insert_Kernel_Function = Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Insert_Kernel_Function;
//     using Search_Kernel_Function = Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Search_Kernel_Function;

//     {
//         size_t n = 65521;
//         size_t seed_sz = 2;
//         size_t m = 1;

//         for (size_t m = 1; m <= 8; m *= 2) 
//         {

//             std::cout << "n: " << n << ", m: " << m << ", seed_sz: " << seed_sz << std::endl;

//                 Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
//                     static_cast<Insert_Kernel_Function *>(insert_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     static_cast<Search_Kernel_Function *>(search_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     n, m, seed_sz);

//                 Unordered_Map_GPU<Key_T, Count_T> emap;
//                 Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));
//                 // test.insert_perf_test(insert_keys_size, 3);
//                 test.search_perf_test(insert_keys_size, search_keys_size);
//                 test.clear();

//                 Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map1(
//                     static_cast<Insert_Kernel_Function *>(insert_warp_test_1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     static_cast<Search_Kernel_Function *>(search_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     n, m, seed_sz);

//                 Test<Key_T, Count_T, Hashed_T> test1(gpu_tool<Key_T>, (map1), (emap));
//                 // test1.insert_perf_test(insert_keys_size, 3);
//                 test1.search_perf_test(insert_keys_size, search_keys_size);
//                 test1.clear();

//                 Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map2(
//                     static_cast<Insert_Kernel_Function *>(insert_warp_test_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     static_cast<Search_Kernel_Function *>(search_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     n, m, seed_sz);

//                 Test<Key_T, Count_T, Hashed_T> test2(gpu_tool<Key_T>, (map2), (emap));
//                 // test2.insert_perf_test(insert_keys_size, 3);
//                 test2.search_perf_test(insert_keys_size, search_keys_size);
//                 test2.clear();


//                 Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map3(
//                     static_cast<Insert_Kernel_Function *>(insert_warp_min<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     static_cast<Search_Kernel_Function *>(search_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     n, m, seed_sz);

//                 Test<Key_T, Count_T, Hashed_T> test3(gpu_tool<Key_T>, (map3), (emap));
//                 // test3.insert_perf_test(insert_keys_size, 3);
//                 test3.search_perf_test(insert_keys_size, search_keys_size);
//                 test3.clear();
//         }
//     }

//     return 0;
//     {
//         size_t n = 65521;
//         size_t seed_sz = 8;

//         for (size_t m = 1; m <= 8; m *= 2) {

//             std::cout << "n: " << n << ", m: " << m << ", seed_sz: " << seed_sz << std::endl;

//                 Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
//                     static_cast<Insert_Kernel_Function *>(insert_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     static_cast<Search_Kernel_Function *>(search_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     n, m, seed_sz);

//                 Unordered_Map_GPU<Key_T, Count_T> emap;
//                 Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));
//                 test.insert_perf_test(insert_keys_size, 3);
//                 test.clear();

//                 Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map1(
//                     static_cast<Insert_Kernel_Function *>(insert_warp_test_1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     static_cast<Search_Kernel_Function *>(search_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     n, m, seed_sz);

//                 Test<Key_T, Count_T, Hashed_T> test1(gpu_tool<Key_T>, (map1), (emap));
//                 test1.insert_perf_test(insert_keys_size, 3);
//                 test1.clear();

//                 Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map2(
//                     static_cast<Insert_Kernel_Function *>(insert_warp_test_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     static_cast<Search_Kernel_Function *>(search_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     n, m, seed_sz);

//                 Test<Key_T, Count_T, Hashed_T> test2(gpu_tool<Key_T>, (map2), (emap));
//                 test2.insert_perf_test(insert_keys_size, 3);
//                 test2.clear();
//         }
//     }


//     {
//         size_t n = 65521;
//         size_t seed_sz = 16;

//         for (size_t m = 1; m <= 8; m *= 2) {

//             std::cout << "n: " << n << ", m: " << m << ", seed_sz: " << seed_sz << std::endl;

//                 Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
//                     static_cast<Insert_Kernel_Function *>(insert_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     static_cast<Search_Kernel_Function *>(search_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     n, m, seed_sz);

//                 Unordered_Map_GPU<Key_T, Count_T> emap;
//                 Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));
//                 test.insert_perf_test(insert_keys_size, 3);
//                 test.clear();

//                 Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map1(
//                     static_cast<Insert_Kernel_Function *>(insert_warp_test_1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     static_cast<Search_Kernel_Function *>(search_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     n, m, seed_sz);

//                 Test<Key_T, Count_T, Hashed_T> test1(gpu_tool<Key_T>, (map1), (emap));
//                 test1.insert_perf_test(insert_keys_size, 3);
//                 test1.clear();

//                 Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map2(
//                     static_cast<Insert_Kernel_Function *>(insert_warp_test_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     static_cast<Search_Kernel_Function *>(search_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//                     n, m, seed_sz);

//                 Test<Key_T, Count_T, Hashed_T> test2(gpu_tool<Key_T>, (map2), (emap));
//                 test2.insert_perf_test(insert_keys_size, 3);
//                 test2.clear();
//         }
//     }


//     return 0;
// }


// int main(int argc, char const *argv[])
// {
//     // cudaDeviceReset();
//     size_t insert_keys_size = 512 * 1024 * 1024;
//     size_t search_keys_size = 32 * 1024 * 1024;
//     // size_t insert_keys_size = 32 * 1024;
//     // size_t search_keys_size = 1024;

//     using Key_T = unsigned int; 
//     using Count_T = unsigned int;
//     using Hashed_T = unsigned int;
//     using Seed_T = unsigned int;

//     using Hash_Function = hash_mul_add<Key_T, Hashed_T, Seed_T>;
//     using Insert_Kernel_Function = Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Insert_Kernel_Function;
//     using Search_Kernel_Function = Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Search_Kernel_Function;

//     Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(
//         static_cast<Insert_Kernel_Function *>(insert_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//         static_cast<Search_Kernel_Function *>(search_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>));

//     Unordered_Map_GPU<Key_T, Count_T> emap;
//     Test<Key_T, Count_T, Hashed_T> test(gpu_tool<Key_T>, (map), (emap));
//     test.insert_perf_test(insert_keys_size, 3);
//     test.clear();

//     Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map1(
//         static_cast<Insert_Kernel_Function *>(insert_warp_test_1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//         static_cast<Search_Kernel_Function *>(search_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>));

//     Test<Key_T, Count_T, Hashed_T> test1(gpu_tool<Key_T>, (map1), (emap));
//     test1.insert_perf_test(insert_keys_size, 3);
//     test1.clear();

//     Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map2(
//         static_cast<Insert_Kernel_Function *>(insert_warp_test_2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//         static_cast<Search_Kernel_Function *>(search_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>));

//     Test<Key_T, Count_T, Hashed_T> test2(gpu_tool<Key_T>, (map2), (emap));
//     test2.insert_perf_test(insert_keys_size, 3);
//     test2.clear();

//     // Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map2(
//     //     static_cast<Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Insert_Kernel_Function *>
//     //         (insert_warp_bit_hash<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>),
//     //     static_cast<Count_Min_Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::Search_Kernel_Function *>
//     //         (search_warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>));
//     // Test<Key_T, Count_T, Hashed_T> test2(gpu_tool<Key_T>, (map2), (emap));
//     // test2.insert_perf_test(insert_keys_size, 3);
//     // test2.clear();

//     // test.clear();
//     // test.search_perf_test(insert_keys_size, search_keys_size, 3);
//     // test.clear();
//     // test.search_accuracy_test(insert_keys_size, search_keys_size);
//     // map.print(debug_log);

//     return 0;
// }

// int main1(int argc, char const *argv[])
// {
//     size_t insert_keys_size = 32 * 1024;
//     size_t search_keys_size = 1024;

//     using Key_T = unsigned int; 
//     using Count_T = unsigned int;

//     // CPU_Tools<Key_T> tool;
//     // Unordered_Map<Key_T, Count_T> map;
//     Count_Min_Sketch_CPU<Key_T, Count_T> map;
//     Unordered_Map<Key_T, Count_T> emap;
//     // Test<unsigned int, unsigned int> test(tool, static_cast<Sketch<Key_T, Count_T> &>(map), static_cast<Sketch<Key_T, Count_T> &>(emap));
//     Test<unsigned int, unsigned int> test(cpu_tool<Key_T>, (map), (emap));

//     // test.insert_perf_test(insert_keys_size, 3);
//     // test.clear();
//     // test.search_perf_test(insert_keys_size, search_keys_size);
//     // test.clear();
//     // test.search_accuracy_test(insert_keys_size, search_keys_size);
//     // map.print(debug_log);
//     return 0;
// }