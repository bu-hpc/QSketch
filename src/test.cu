#include <qsketch.h>

using Key_T = unsigned int; 
using Count_T = unsigned int;
using Hashed_T = unsigned int;
using Seed_T = unsigned int;

using Hash_Function = qsketch::hash_mul_add<Key_T, Hashed_T, Seed_T>;

// debug
// size_t insert_keys_sz = 32;
// size_t search_keys_sz = 8;
// size_t insert_keys_sz = 128 *1024;
// size_t search_keys_sz = 128 * 128;

// // small
// size_t insert_keys_sz = 1024 * 1024;
// size_t search_keys_sz = 128 * 1024;
// // //size_t n = 8191;
// size_t n = 65521;
// size_t n_sub_warp = 262139;
// size_t n = 1024 * 1024;

size_t insert_keys_sz = 4 * 1024 * 1024;
size_t search_keys_sz = 1024 * 1024;

// // // medium
// size_t insert_keys_sz = 32 * 1024 * 1024;
// size_t search_keys_sz = 8 * 1024 * 1024;
// // size_t search_keys_sz = 1024 * 1024;
// // size_t n_warp = 131071;
// // size_t n = 2097143;
// size_t n_sub_warp = 8388593;

// // large
// size_t insert_keys_sz = 128 * 1024 * 1024;
// size_t search_keys_sz = 128 * 1024 * 1024;
// // //size_t n = 1048573; //  < 1024 * 1024
// // // size_t n = 8388593;
// // // size_t n_sub_warp = 33554393;
// // // size_t n = 33554393;
// size_t n = 128 * 1024 * 1024;

// size_t insert_keys_sz = 134217728;
// size_t insert_keys_sz = 1024;
// size_t search_keys_sz = 32;



// size_t insert_loop = 1;
// size_t search_loop = 1;

// size_t n = 8388593;
// size_t n = 7456549;
// size_t n = 257731;
// size_t n = 1048573; //  < 1024 * 1024
// size_t n = 65521;
// size_t n = 97;
size_t seed_sz = 1;
size_t m = 3;
// size_t m = 1;


int sf_host_sub_warp_test_8_fly() {
    dim3 gridDim(1, 1, 1);
    int times = 8;
    size_t sub_warp_m = 8;
    
    std::cout << "sf_host_sub_warp_test_8_fly" << std::endl;
    
    size_t n = qsketch::Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::number_of_buckets(insert_keys_sz,
        sub_warp_m, 0.5);
        // size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, sub_warp_m);
    // n = 33554393;
    // n = 33554432;
    // n = 1023;
    std::cout << "n: " << n << std::endl;
    std::cout << "insert_keys_sz: " << insert_keys_sz << "," << n; std::cout << std::endl;
    // n = 1048573;
    // n = 2097143;
    // n = 32 * 1024 * 1024;
    qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
    qsketch::Sketch_GPU_SF_Host_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
        sub_warp_m, 8, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);
    // map.gridDim = gridDim;
    // dim3 &blockDim = map.blockDim;
    // map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    // map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);
    
    qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
    double ip = test.insert_perf_test(insert_keys_sz, std::cerr);
    std::cout << "sub_warp p1: " << ip << std::endl;
    // test.clear();
    double sp = test.search_perf_test(search_keys_sz, std::cerr);
    std::cout << "sub_warp p2: " << sp << std::endl;
    test.clear();
    qsketch::check_cuda_error();
    // test.search_accuracy_test(insert_keys_sz, search_keys_sz);
    test.clear(); 
    return 0;
}


int sf_host_sub_warp_test_8() {
    // dim3 gridDim(1, 1, 1);
    
    std::cout << "sf_host_sub_warp_test_8" << std::endl;
    // size_t sub_warp_m = 8;
    size_t sub_warp_m = 8;
    size_t n = qsketch::Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::number_of_buckets(insert_keys_sz,
        sub_warp_m, 0.5);
        // size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, sub_warp_m);
    // n = 33554393;
    // n = 33554432;
    // n = 1023;
    std::cout << "n: " << n << std::endl;
    std::cout << "insert_keys_sz: " << insert_keys_sz << "," << n; std::cout << std::endl;
    // n = 1048573;
    // n = 2097143;
    // n = 32 * 1024 * 1024;
    qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
    qsketch::Sketch_GPU_SF_Host_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
        sub_warp_m, 8, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);
    // map.gridDim = gridDim;
    // dim3 &blockDim = map.blockDim;
    // map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    // map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);
    
    qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
    // double ip = test.insert_perf_test(insert_keys_sz, std::cerr);
    // std::cout << "sub_warp p1: " << ip << std::endl;
    // // test.clear();
    // double sp = test.search_perf_test(search_keys_sz, std::cerr);
    // std::cout << "sub_warp p2: " << sp << std::endl;
    // test.clear();
    qsketch::check_cuda_error();
    // test.search_accuracy_test(insert_keys_sz, search_keys_sz);
    test.clear(); 

    return 0;
}


int sf_managed_sub_warp_test_8() {
    // dim3 gridDim(1, 1, 1);
    
    std::cout << "sf_sub_warp_test_8" << std::endl;
    // size_t sub_warp_m = 8;
    size_t sub_warp_m = 8;
    size_t n = qsketch::Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::number_of_buckets(insert_keys_sz,
        sub_warp_m, 0.5);
        // size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, sub_warp_m);
    // n = 33554393;
    // n = 33554432;
    // n = 1023;
    std::cout << "n: " << n << std::endl;
    std::cout << "insert_keys_sz: " << insert_keys_sz << "," << n; std::cout << std::endl;
    // n = 1048573;
    // n = 2097143;
    // n = 32 * 1024 * 1024;
    qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
    qsketch::Sketch_GPU_SF_Managed_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
        sub_warp_m, 8, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);
    // map.gridDim = gridDim;
    // dim3 &blockDim = map.blockDim;
    // map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    // map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);
    
    qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
    // double ip = test.insert_perf_test(insert_keys_sz, std::cerr);
    // std::cout << "sub_warp p1: " << ip << std::endl;
    // // test.clear();
    // double sp = test.search_perf_test(search_keys_sz, std::cerr);
    // std::cout << "sub_warp p2: " << sp << std::endl;
    // test.clear();
    qsketch::check_cuda_error();
    test.search_accuracy_test(insert_keys_sz, search_keys_sz);
    test.clear(); 

    return 0;
}

int sf_sub_warp_test_8() {
    // dim3 gridDim(1, 1, 1);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("cuda flag: %d\n", int(prop.canMapHostMemory));
        if (!prop.canMapHostMemory) 
            exit(0);
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaDeviceSynchronize();

    std::cout << "sf_sub_warp_test_8" << std::endl;
    // size_t sub_warp_m = 8;
    size_t sub_warp_m = 8;
    size_t n = qsketch::Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::number_of_buckets(insert_keys_sz,
        sub_warp_m, 0.5);
        // size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, sub_warp_m);
    // n = 33554393;
    // n = 33554432;
    // n = 1023;
    std::cout << "n: " << n << std::endl;
    std::cout << "insert_keys_sz: " << insert_keys_sz << "," << n; std::cout << std::endl;
    // n = 1048573;
    // n = 2097143;
    // n = 32 * 1024 * 1024;
    qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
    qsketch::Sketch_GPU_SF_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
        sub_warp_m, 8, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);
    // map.gridDim = gridDim;
    // dim3 &blockDim = map.blockDim;
    // map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    // map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);
    
    qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
    // double ip = test.insert_perf_test(insert_keys_sz, std::cerr);
    // std::cout << "sub_warp p1: " << ip << std::endl;
    // // test.clear();
    // double sp = test.search_perf_test(search_keys_sz, std::cerr);
    // std::cout << "sub_warp p2: " << sp << std::endl;
    // test.clear();
    qsketch::check_cuda_error();
    test.search_accuracy_test(insert_keys_sz, search_keys_sz);
    test.clear(); 

    return 0;
}

// int level_sub_warp1_test_8() {
//     size_t insert_keys_sz = 1024 * 1024;
//     size_t search_keys_sz = 1024 * 1024;
    
//     std::cout << "level_sub_warp_test111111111111111111_8" << std::endl;
//     size_t sub_warp_m = 8;
//     // size_t sub_warp_m = 16;
//     size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, sub_warp_m);
//     // n = 33554393;
//     // n = 33554432;
//     n = 262139;
//     std::cout << "n: " << n << std::endl;
//     // n = 1048573;
//     // n = 2097143;
//     // n = 32 * 1024 * 1024;
//     qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
//     qsketch::Sketch_GPU_Sub_Warp_Level1<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
//         sub_warp_m, 1024, qsketch::default_values::WARP_SIZE, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);

//     // dim3 gridDim(1, 1, 1);
//     // map.gridDim = gridDim;
//     // dim3 &blockDim = map.blockDim;
//     // map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
//     // map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);

//     qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
//     // double ip = test.insert_perf_test(insert_keys_sz, std::cerr);
//     // std::cout << "level_sub_warp p1: " << ip << std::endl;
//     // test.clear();

//     // test.insert_perf_test_old(insert_keys_sz, 1, std::cerr);
//     // test.clear();

//     // double sp = test.search_perf_test(search_keys_sz, std::cerr);
//     // std::cout << "level_sub_warp p2: " << sp << std::endl;
//     // test.clear();
    
//     // test.search_perf_test_old(insert_keys_sz, search_keys_sz, 1, std::cerr);
//     // test.clear();

//     std::cout << "----------------------------------------" << std::endl;
//     test.insert_perf_freq_test_old(insert_keys_sz, 1);
//     test.clear();
//     test.search_perf_freq_test_old(insert_keys_sz, search_keys_sz, 1);
//     test.clear();

//     // qsketch::check_cuda_error();
//     // test.search_accuracy_test(insert_keys_sz, search_keys_sz);
//     // test.clear(); 


//     return 0;
// }

int level_sub_warp_lock_test_8() {
    // size_t insert_keys_sz = 1024 * 1024;
    // size_t search_keys_sz = 128 * 1024;
    // size_t search_keys_sz = 1;
    
    std::cout << "level_sub_warp_test_8" << std::endl;
    size_t sub_warp_m = 8;
    // size_t sub_warp_m = 16;
    size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, sub_warp_m);
    // n = 33554393;
    // n = 33554432;
    // n = 262139;
    std::cout << "n: " << n << std::endl;
    // n = 1048573;
    // n = 2097143;
    // n = 32 * 1024 * 1024;
    qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
    qsketch::Sketch_GPU_Sub_Lock_Warp_Level<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
        sub_warp_m, 1024, qsketch::default_values::WARP_SIZE, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);

    // dim3 gridDim(1, 1, 1);
    // map.gridDim = gridDim;
    // dim3 &blockDim = map.blockDim;
    // map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    // map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);

    qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
    double ip = test.insert_perf_test(insert_keys_sz, std::cerr);
    std::cout << "level_sub_warp p1: " << ip << std::endl;
    test.clear();

    // test.insert_perf_test_old(insert_keys_sz, 1, std::cerr);
    // test.clear();

    double sp = test.search_perf_test(search_keys_sz, std::cerr);
    std::cout << "level_sub_warp p2: " << sp << std::endl;
    test.clear();
    
    // test.search_perf_test_old(insert_keys_sz, search_keys_sz, 1, std::cerr);
    // test.clear();

    // std::cout << "----------------------------------------" << std::endl;
    // test.insert_perf_freq_test_old(insert_keys_sz, 1);
    // test.clear();
    // test.search_perf_freq_test_old(insert_keys_sz, search_keys_sz, 1);
    // test.clear();

    // qsketch::check_cuda_error();
    test.search_accuracy_test(insert_keys_sz, search_keys_sz);
    test.clear(); 


    return 0;
}


int level_sub_warp_test_8() {
    size_t insert_keys_sz = 1024 * 1024;
    // size_t search_keys_sz = 128 * 1024;
    size_t search_keys_sz = 1;
    
    std::cout << "level_sub_warp_test_8" << std::endl;
    size_t sub_warp_m = 8;
    // size_t sub_warp_m = 16;
    size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, sub_warp_m);
    // n = 33554393;
    // n = 33554432;
    n = 262139;
    std::cout << "n: " << n << std::endl;
    // n = 1048573;
    // n = 2097143;
    // n = 32 * 1024 * 1024;
    qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
    qsketch::Sketch_GPU_Sub_Warp_Level<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
        sub_warp_m, 1024, qsketch::default_values::WARP_SIZE, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);

    // dim3 gridDim(1, 1, 1);
    // map.gridDim = gridDim;
    // dim3 &blockDim = map.blockDim;
    // map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    // map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);

    qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
    // double ip = test.insert_perf_test(insert_keys_sz, std::cerr);
    // std::cout << "level_sub_warp p1: " << ip << std::endl;
    // test.clear();

    // test.insert_perf_test_old(insert_keys_sz, 1, std::cerr);
    // test.clear();

    // double sp = test.search_perf_test(search_keys_sz, std::cerr);
    // std::cout << "level_sub_warp p2: " << sp << std::endl;
    // test.clear();
    
    // test.search_perf_test_old(insert_keys_sz, search_keys_sz, 1, std::cerr);
    // test.clear();

    // std::cout << "----------------------------------------" << std::endl;
    // test.insert_perf_freq_test_old(insert_keys_sz, 1);
    // test.clear();
    // test.search_perf_freq_test_old(insert_keys_sz, search_keys_sz, 1);
    // test.clear();

    // qsketch::check_cuda_error();
    test.search_accuracy_test(insert_keys_sz, search_keys_sz);
    test.clear(); 


    return 0;
}


int sub_warp_test_8() {
    std::cout << "sub_warp_test_8" << std::endl;
    // size_t sub_warp_m = 8;
    size_t sub_warp_m = 8;
    size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, sub_warp_m);
    // n = 33554393;
    // n = 33554432;
    // std::cout << "n: " << n << std::endl;
    std::cout << "insert_keys_sz: " << insert_keys_sz << "," << n; std::cout << std::endl;
    // n = 1048573;
    // n = 2097143;
    // n = 32 * 1024 * 1024;
    qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
    qsketch::Sketch_GPU_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
        sub_warp_m, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);

    qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
    double ip = test.insert_perf_test(insert_keys_sz, std::cerr);
    // std::cout << "sub_warp p1: " << ip << std::endl;
    // test.clear();
    double sp = test.search_perf_test(search_keys_sz, std::cerr);
    // std::cout << "sub_warp p2: " << sp << std::endl;
    test.clear();
    // qsketch::check_cuda_error();
    // test.search_accuracy_test(insert_keys_sz, search_keys_sz);
    // test.clear(); 

    return 0;
}

int sub_warp_test_16() {
    std::cout << "sub_warp_test_16" << std::endl;
    // size_t sub_warp_m = 8;
    size_t sub_warp_m = 16;
    size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, sub_warp_m);
    n = 33554393;
    std::cout << "n: " << n << std::endl;
    // n = 1048573;
    // n = 2097143;
    // n = 32 * 1024 * 1024;
    qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
    qsketch::Sketch_GPU_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
        sub_warp_m, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);

    qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
    test.insert_perf_test(insert_keys_sz, std::cerr);
    std::cout << "sub_warp p1" << std::endl;
    // test.clear();
    test.search_perf_test(search_keys_sz, std::cerr);
    std::cout << "sub_warp p2" << std::endl;
    test.clear();

    test.search_accuracy_test(insert_keys_sz, search_keys_sz);
    test.clear(); 

    return 0;
}

int warp_test_8() {
    std::cout << "warp_test_8" << std::endl;
    size_t sub_warp_m = 8;
    size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, sub_warp_m);
    std::cout << "n: " << n << std::endl;
    // n = 1048573;
    // n = 2097143;
    // n = 32 * 1024 * 1024;
    qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
    qsketch::Sketch_GPU_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
        sub_warp_m, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);

    dim3 gridDim(65536, 1, 1);
    map.gridDim = gridDim;
    dim3 &blockDim = map.blockDim;
    blockDim = dim3(8, 1, 1);
    map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);

    qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
    test.insert_perf_test(insert_keys_sz, std::cerr);
    // std::cout << "warp p1" << std::endl;
    // test.clear();
    test.search_perf_test(search_keys_sz, std::cerr);
    // std::cout << "warp p2" << std::endl;
    test.clear();

    test.search_accuracy_test(insert_keys_sz, search_keys_sz);
    test.clear(); 

    return 0;
}

int warp_test() {
    std::cout << "warp" << std::endl;
    size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, qsketch::default_values::WARP_SIZE);
    std::cout << "n: " << n << std::endl;
    // n = 1048573;
    // n = 2097143;
    // n = 32 * 1024 * 1024;
    qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
    qsketch::Sketch_GPU_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
        qsketch::default_values::WARP_SIZE, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);

    qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
    test.insert_perf_test(insert_keys_sz, std::cerr);
    // std::cout << "warp p1" << std::endl;
    // test.clear();
    test.search_perf_test(search_keys_sz, std::cerr);
    // std::cout << "warp p2" << std::endl;
    test.clear();

    test.search_accuracy_test(insert_keys_sz, search_keys_sz);
    test.clear(); 

    return 0;
}

int warm() {
    // {
    //     qsketch::Unordered_Map<Key_T, Count_T> emap;
    //     Key_T *keys = new Key_T[128 * 128];
    //     size_t keys_sz = 0;
    //     for (uint i = 1; i <= 128; ++i) {
    //         for (uint j = i; j <= 128; ++j) {
    //             keys[keys_sz] = j;
    //             keys_sz ++;
    //         }
    //     }
    //     Count_T *counts = new Count_T[128 * 128];
    //     emap.insert(keys, keys_sz);
    //     size_t rt = emap.get_counts(keys, keys_sz, counts);

    //     for (uint i = 0; i < rt; ++i) {
    //         std::cout << i << "\t:" << keys[i] << "\t\t<----->\t\t" << counts[i] << std::endl;
    //     }
    //     std::cout << "rt: " << rt << std::endl;

    // }
    // return 0;
    std::cout << "warm" << std::endl;
    size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, m);
    std::cout << "n: " << n << std::endl;
    // n = 11184799;
    // n = 22369601;
    qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
    // std::cout << "warm p0" << std::endl;
    qsketch::Sketch_GPU_Thread<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n, m, seed_sz);
    // qsketch::Unordered_Map_GPU<Key_T, Count_T> map;
    // std::cout << "warm p1" << std::endl;
    qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
    // std::cout << "warm p2" << std::endl;
    test.insert_perf_test(insert_keys_sz, std::cerr);
    test.clear();
    // std::cout << "warm p3" << std::endl;
    test.search_perf_test(search_keys_sz, std::cerr);
    test.clear();
    // std::cout << "warm p4" << std::endl;
    // test.search_accuracy_test(insert_keys_sz, search_keys_sz);
    // test.clear(); 
    // test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz, 1);
    // test.clear(); 

    return 0;
}

int main(int argc, char const *argv[])
{

    // size_t keys_sz = 25165716;
    // unsigned int *keys;
    // // CUDA_CALL(cudaMalloc(&keys, sizeof(unsigned int) * keys_sz));
    // std::cout << "p1" << std::endl;
    // cudaMalloc(&keys, sizeof(unsigned int) * keys_sz);
    // std::cout << "p2" << std::endl;
    // cudaMemset(keys, 0, sizeof(unsigned int) * keys_sz);
    // std::cout << "p3" << std::endl;
    // cudaDeviceSynchronize();

    warm();
    // warp_test();
    // warp_test_8();
    // sub_warp_test_8();
    // sub_warp_test_16();    
    // level_sub_warp_test_8();
    // level_sub_warp1_test_8();
    // sf_sub_warp_test_8();
    // sf_managed_sub_warp_test_8();
    // sf_host_sub_warp_test_8();

    // level_sub_warp_lock_test_8();
    sf_host_sub_warp_test_8_fly();
    return 0;
}

