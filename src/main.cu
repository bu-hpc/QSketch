#include <qsketch.h>

using Key_T = unsigned int; 
using Count_T = unsigned int;
using Hashed_T = unsigned int;
using Seed_T = unsigned int;

using Hash_Function = qsketch::hash_mul_add<Key_T, Hashed_T, Seed_T>;


size_t seed_sz = 1;
size_t m = 3;

// int sub_warp_test() {
//     std::cout << "sub_warp" << std::endl;
//     size_t sub_warp_m = 8;
//     size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, sub_warp_m);
//     std::cout << "n: " << n << std::endl;
//     // n = 1048573;
//     // n = 2097143;
//     // n = 32 * 1024 * 1024;
//     qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
//     qsketch::Sketch_GPU_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
//         sub_warp_m, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);

//     qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
//     // test.insert_perf_test(insert_keys_sz, std::cerr);
//     // std::cout << "sub_warp p1" << std::endl;
//     // // test.clear();
//     // test.search_perf_test(search_keys_sz, std::cerr);
//     // std::cout << "sub_warp p2" << std::endl;
//     // test.clear();

//     test.search_accuracy_test(insert_keys_sz, search_keys_sz);
//     test.clear(); 

//     return 0;
// }

// int warp_test() {
//     std::cout << "warp" << std::endl;
//     size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, qsketch::default_values::WARP_SIZE);
//     std::cout << "n: " << n << std::endl;
//     // n = 1048573;
//     // n = 2097143;
//     // n = 32 * 1024 * 1024;
//     qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
//     qsketch::Sketch_GPU_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
//         qsketch::default_values::WARP_SIZE, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);

//     qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
//     test.insert_perf_test(insert_keys_sz, std::cerr);
//     // std::cout << "warp p1" << std::endl;
//     // test.clear();
//     test.search_perf_test(search_keys_sz, std::cerr);
//     // std::cout << "warp p2" << std::endl;
//     test.clear();

//     // test.search_accuracy_test(insert_keys_sz, search_keys_sz);
//     test.clear(); 

//     return 0;
// }

int warm(size_t insert_keys_sz) {

    std::cout << "warm" << std::endl;
    size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, m);
    // std::cout << "n: " << n << std::endl;
    
    qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
    qsketch::Sketch_GPU_Thread<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n, m, seed_sz);
    qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
    test.insert_perf_test(insert_keys_sz, std::cerr);
    // test.search_accuracy_test(insert_keys_sz, search_keys_sz);
    test.clear(); 
    return 0;
}

// size_t iks_start = 32 * 1024; // iks  : insert_keys_sz
size_t iks_start = 32 * 1024 * 1024;
// size_t iks_start = 32 * 1024 * 1024;

size_t iks_end = 128 * 1024 * 1024;
// size_t iks_end = 32 * 1024 * 1024;
// size_t iks_end = 16 * 1024 * 1024;
// size_t iks_end = 32 * 1024;


// size_t dimx_start = 128;
// size_t dimx_end = 128;
// size_t dimx_start = 128;//32; // 16
// size_t dimx_end = 32; // 8192
size_t dimx_start = 16;
size_t dimx_end = 8192;

bool insert_search_test = true;
bool search_accuracy_test = false;
// bool insert_search_test = false;
// bool search_accuracy_test = true;
bool search_accuracy_freq_test = false;

int thread_test = 1;
int thread_test2 = 0;
int warp_test = 0;
int warp_test_8 = 0;
int sub_warp_test = 0;
int sub_warp_level_test = 0;
int sub_warp_level_lock_test = 0;
int sf_test = 0;
int sf_host_test = 0;

double work_load_factor = 0.5;


void run_test(double &max_insert_perf, double &max_search_perf,
    size_t insert_keys_sz, size_t search_keys_sz, 
    qsketch::Sketch_GPU<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> &map,
    qsketch::Test<Key_T, Count_T> &test) {
    if (insert_search_test) {
        double t_insert_perf = test.insert_perf_test(insert_keys_sz, std::cout);
        double t_search_perf = test.search_perf_test(search_keys_sz, std::cout);
        max_insert_perf = std::max(max_insert_perf, t_insert_perf);
        max_search_perf = std::max(max_search_perf, t_search_perf);
        test.clear();
        // std::cout << "t: " << t_insert_perf << "," << t_search_perf << std::endl;
    }
    
    // std::cout << t_insert_perf << ";" << std::endl;
    
    if (search_accuracy_test) {
        double accracy = test.search_accuracy_test(insert_keys_sz, search_keys_sz);
        test.clear(); 
        // std::cout << accracy << std::endl;
        std::cout << accracy << ", ";
        map.show_memory_usage(std::cout);
    }

    if (search_accuracy_freq_test) {
        double accracy = test.search_accuracy_freq_test(insert_keys_sz, search_keys_sz);
        test.clear(); 
        // std::cout << accracy << std::endl;
        // std::cout << accracy << ", ";
        map.show_memory_usage(std::cout);
    }
}

int main1()
{
    warm(1024 * 1024);
    std::cout << "work_load_factor: " << work_load_factor << std::endl;

    if(thread_test){
        std::cout << "Thread: " << std::endl;
    
        // size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, m);
        for (size_t insert_keys_sz = iks_start; insert_keys_sz <= iks_end; insert_keys_sz *= 2) 
        {
        // size_t insert_keys_sz = 128 * 1024 * 1024; {
            // std::cout << insert_keys_sz << ","; std::cout << std::endl;
            size_t n = qsketch::Sketch_GPU_Thread<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::number_of_buckets(insert_keys_sz, m, work_load_factor);
            size_t search_keys_sz =  insert_keys_sz / 2;
            double max_insert_perf = 0.0;
            double max_search_perf = 0.0;
            for (size_t dimx = dimx_start; dimx <= dimx_end; dimx *= 2) {
                dim3 gridDim(insert_keys_sz / dimx, 1, 1);
                // std::cout << "p1" << std::endl;
                // std::cout << gridDim.x << ",";
                
                qsketch::Sketch_GPU_Thread<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n, m, seed_sz);
                map.gridDim = gridDim;
                dim3 &blockDim = map.blockDim;
                map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
                map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);
                qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
                qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
                
                run_test(max_insert_perf, max_search_perf, insert_keys_sz, search_keys_sz, map, test);

            }
            if (insert_search_test) {
                std::cout << max_insert_perf << "," << max_search_perf << std::endl;
            }
            
            // std::cout << std::endl;
        }    
    }
    std::cout << "----------------------------------------" << std::endl;
    if(thread_test2){
        std::cout << "Thread 32: " << std::endl;
        size_t sub_warp_m = 32;
        // size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, m);
        for (size_t insert_keys_sz = iks_start; insert_keys_sz <= iks_end; insert_keys_sz *= 2) 
        {
        // size_t insert_keys_sz = 128 * 1024 * 1024; {
            // std::cout << insert_keys_sz << ","; std::cout << std::endl;
            size_t n = qsketch::Sketch_GPU_Thread2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::number_of_buckets(insert_keys_sz, sub_warp_m, work_load_factor);
            size_t search_keys_sz =  insert_keys_sz / 2;
            double max_insert_perf = 0.0;
            double max_search_perf = 0.0;
            for (size_t dimx = dimx_start; dimx <= dimx_end; dimx *= 2) {
                dim3 gridDim(insert_keys_sz / dimx, 1, 1);
                // std::cout << "p1" << std::endl;
                // std::cout << gridDim.x << ",";
                
                qsketch::Sketch_GPU_Thread2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n, sub_warp_m, seed_sz);
                map.gridDim = gridDim;
                dim3 &blockDim = map.blockDim;
                map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
                map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);
                qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
                qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
                
                run_test(max_insert_perf, max_search_perf, insert_keys_sz, search_keys_sz, map, test);

            }
            if (insert_search_test) {
                std::cout << max_insert_perf << "," << max_search_perf << std::endl;
            }
            
            // std::cout << std::endl;
        }    
    }
    std::cout << "----------------------------------------" << std::endl;

        if(thread_test2){
        std::cout << "Thread 8: " << std::endl;
        size_t sub_warp_m = 8;
        // size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, m);
        for (size_t insert_keys_sz = iks_start; insert_keys_sz <= iks_end; insert_keys_sz *= 2) 
        {
        // size_t insert_keys_sz = 128 * 1024 * 1024; {
            // std::cout << insert_keys_sz << ","; std::cout << std::endl;
            size_t n = qsketch::Sketch_GPU_Thread2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::number_of_buckets(insert_keys_sz, sub_warp_m, work_load_factor);
            size_t search_keys_sz =  insert_keys_sz / 2;
            double max_insert_perf = 0.0;
            double max_search_perf = 0.0;
            for (size_t dimx = dimx_start; dimx <= dimx_end; dimx *= 2) {
                dim3 gridDim(insert_keys_sz / dimx, 1, 1);
                // std::cout << "p1" << std::endl;
                // std::cout << gridDim.x << ",";
                
                qsketch::Sketch_GPU_Thread2<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n, sub_warp_m, seed_sz);
                map.gridDim = gridDim;
                dim3 &blockDim = map.blockDim;
                map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
                map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);
                qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
                qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
                
                run_test(max_insert_perf, max_search_perf, insert_keys_sz, search_keys_sz, map, test);

            }
            if (insert_search_test) {
                std::cout << max_insert_perf << "," << max_search_perf << std::endl;
            }
            
            // std::cout << std::endl;
        }    
    }
    std::cout << "----------------------------------------" << std::endl;

    if(warp_test){
        std::cout << "Warp: " << std::endl;
        // size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, qsketch::default_values::WARP_SIZE);
        for (size_t insert_keys_sz = iks_start; insert_keys_sz <= iks_end; insert_keys_sz *= 2) 
        {
            // std::cout << insert_keys_sz << ","; std::cout << std::endl;
            size_t n = qsketch::Sketch_GPU_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::number_of_buckets
                (insert_keys_sz, qsketch::default_values::WARP_SIZE, work_load_factor);
            size_t search_keys_sz =  insert_keys_sz / 2;
            double max_insert_perf = 0.0;
            double max_search_perf = 0.0;
            for (size_t dimx = dimx_start; dimx <= dimx_end; dimx *= 2) {
                dim3 gridDim(insert_keys_sz / dimx, 1, 1);
                // std::cout << gridDim.x << ",";
                qsketch::Sketch_GPU_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
                    qsketch::default_values::WARP_SIZE, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);
                map.gridDim = gridDim;
                dim3 &blockDim = map.blockDim;
                map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
                map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);
                qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
                qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
                
                run_test(max_insert_perf, max_search_perf, insert_keys_sz, search_keys_sz, map, test);
            }
            if (insert_search_test) {
                std::cout << max_insert_perf << "," << max_search_perf << std::endl;
            }
        }    
    }

    std::cout << "----------------------------------------" << std::endl;


    if(warp_test_8){
        std::cout << "Warp 8: " << std::endl;
        size_t sub_warp_m = 8;
        for (size_t insert_keys_sz = iks_start; insert_keys_sz <= iks_end; insert_keys_sz *= 2) 
        {
            // std::cout << insert_keys_sz << ","; std::cout << std::endl;
            size_t n = qsketch::Sketch_GPU_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::number_of_buckets
                (insert_keys_sz, sub_warp_m, work_load_factor);
            size_t search_keys_sz =  insert_keys_sz / 2;
            double max_insert_perf = 0.0;
            double max_search_perf = 0.0;
            for (size_t dimx = dimx_start; dimx <= dimx_end; dimx *= 2) {
                dim3 gridDim(insert_keys_sz / dimx, 1, 1);
                // std::cout << gridDim.x << ",";
                qsketch::Sketch_GPU_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
                    sub_warp_m, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);
                map.gridDim = gridDim;
                dim3 &blockDim = map.blockDim;
                blockDim = dim3(8, 1, 1);
                map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
                map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);
                qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
                qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
                
                run_test(max_insert_perf, max_search_perf, insert_keys_sz, search_keys_sz, map, test);
            }
            if (insert_search_test) {
                std::cout << max_insert_perf << "," << max_search_perf << std::endl;
            }
        }    
    }

    std::cout << "----------------------------------------" << std::endl;

    if(sub_warp_test){
        std::cout << "Sub_Warp: " << std::endl;
        size_t sub_warp_m = 8;
        for (size_t insert_keys_sz = iks_start; insert_keys_sz <= iks_end; insert_keys_sz *= 2) 
        // size_t insert_keys_sz = 32 * 1024 * 1024;
        {
            // size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, sub_warp_m);
                // std::cout << n << std::endl;
            // std::cout << insert_keys_sz << ","; std::cout << std::endl;
            size_t n = qsketch::Sketch_GPU_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::number_of_buckets
                (insert_keys_sz, sub_warp_m, work_load_factor);
            size_t search_keys_sz =  insert_keys_sz / 4;
            double max_insert_perf = 0.0;
            double max_search_perf = 0.0;
            for (size_t dimx = dimx_start; dimx <= dimx_end; dimx *= 2) {
                dim3 gridDim(insert_keys_sz / dimx, 1, 1);
                // std::cout << gridDim.x << ",";

                // n = 33554393;
                qsketch::Sketch_GPU_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
                    sub_warp_m, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);
                map.gridDim = gridDim;
                dim3 &blockDim = map.blockDim;
                map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
                map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);
                qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
                qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
                
                run_test(max_insert_perf, max_search_perf, insert_keys_sz, search_keys_sz, map, test);

            }
            if (insert_search_test) {
                std::cout << max_insert_perf << "," << max_search_perf << std::endl;
            }
        }    
    }

    std::cout << "----------------------------------------" << std::endl;
    if(sub_warp_level_test){
        std::cout << "Sub_Warp_Level: " << std::endl;
        size_t sub_warp_m = 8;
        for (size_t insert_keys_sz = iks_start; insert_keys_sz <= iks_end; insert_keys_sz *= 2)
        // size_t insert_keys_sz = 32 * 1024 * 1024;
        {
            // size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, sub_warp_m);
                // std::cout << n << std::endl;
            // std::cout << insert_keys_sz << ","; std::cout << std::endl;
            size_t n = qsketch::Sketch_GPU_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::number_of_buckets
                (insert_keys_sz, sub_warp_m, work_load_factor);
            size_t search_keys_sz =  insert_keys_sz / 4;
            double max_insert_perf = 0.0;
            double max_search_perf = 0.0;
            for (size_t dimx = dimx_start; dimx <= dimx_end; dimx *= 2) {
                dim3 gridDim(insert_keys_sz / dimx, 1, 1);
                // std::cout << gridDim.x << ",";

                // n = 33554393;
                // qsketch::Sketch_GPU_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
                //     sub_warp_m, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);
                qsketch::Sketch_GPU_Sub_Warp_Level<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
                    sub_warp_m, 1024, qsketch::default_values::WARP_SIZE, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);
                map.gridDim = gridDim;
                dim3 &blockDim = map.blockDim;
                map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
                map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);
                qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
                qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
                
                run_test(max_insert_perf, max_search_perf, insert_keys_sz, search_keys_sz, map, test);

            }
            if (insert_search_test) {
                std::cout << max_insert_perf << "," << max_search_perf << std::endl;
            }
        }    
    }
    std::cout << "----------------------------------------" << std::endl;
    if(sub_warp_level_lock_test){
        std::cout << "Sub_Lock_Warp_Level: " << std::endl;
        size_t sub_warp_m = 8;
        for (size_t insert_keys_sz = iks_start; insert_keys_sz <= iks_end; insert_keys_sz *= 2)
        // size_t insert_keys_sz = 32 * 1024 * 1024;
        {
            // size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, sub_warp_m);
                // std::cout << n << std::endl;
            // std::cout << insert_keys_sz << ","; std::cout << std::endl;
            size_t n = qsketch::Sketch_GPU_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::number_of_buckets
                (insert_keys_sz, sub_warp_m, work_load_factor);
            size_t search_keys_sz =  insert_keys_sz / 4;
            double max_insert_perf = 0.0;
            double max_search_perf = 0.0;
            for (size_t dimx = dimx_start; dimx <= dimx_end; dimx *= 2) {
                dim3 gridDim(insert_keys_sz / dimx, 1, 1);
                // std::cout << gridDim.x << ",";

                // n = 33554393;
                // qsketch::Sketch_GPU_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
                //     sub_warp_m, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);
                qsketch::Sketch_GPU_Sub_Lock_Warp_Level<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
                    sub_warp_m, 1024, qsketch::default_values::WARP_SIZE, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);
                map.gridDim = gridDim;
                dim3 &blockDim = map.blockDim;
                map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
                map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);
                qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
                qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
                
                // map.dht_low.print_next_level_id();
                run_test(max_insert_perf, max_search_perf, insert_keys_sz, search_keys_sz, map, test);

            }
            if (insert_search_test) {
                std::cout << max_insert_perf << "," << max_search_perf << std::endl;
            }
        }    
    }
    std::cout << "----------------------------------------" << std::endl;
    if(sf_test){
        int times = 8;
        size_t sub_warp_m = 8;
        std::cout << "SF_Sub_Warp: " << "times: " << times << ", sub_warp_m: " << sub_warp_m << std::endl;
        // size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, qsketch::default_values::WARP_SIZE);
        for (size_t insert_keys_sz = iks_start; insert_keys_sz <= iks_end; insert_keys_sz *= 2) 
        {
            // std::cout << insert_keys_sz << ","; std::cout << std::endl;
            size_t n = qsketch::Sketch_GPU_SF_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::number_of_buckets
                (insert_keys_sz, sub_warp_m, work_load_factor);
            size_t search_keys_sz =  insert_keys_sz / 2;
            double max_insert_perf = 0.0;
            double max_search_perf = 0.0;
            for (size_t dimx = dimx_start; dimx <= dimx_end; dimx *= 2) {
                dim3 gridDim(insert_keys_sz / dimx, 1, 1);
                // std::cout << gridDim.x << ",";
                qsketch::Sketch_GPU_SF_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
                    sub_warp_m, times, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);
                map.gridDim = gridDim;
                dim3 &blockDim = map.blockDim;
                map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
                map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);
                qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
                qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
                
                run_test(max_insert_perf, max_search_perf, insert_keys_sz, search_keys_sz, map, test);
            }
            if (insert_search_test) {
                std::cout << max_insert_perf << "," << max_search_perf << std::endl;
            }
        }    
    }

    std::cout << "----------------------------------------" << std::endl;

    if(sf_host_test){
        int times = 3;
        size_t sub_warp_m = 8;
        std::cout << "SF_Host_Sub_Warp: " << "times: " << times << ", sub_warp_m: " << sub_warp_m << std::endl;
        // size_t n = qsketch::calculator::number_of_buckets(insert_keys_sz, qsketch::default_values::WARP_SIZE);
        for (size_t insert_keys_sz = iks_start; insert_keys_sz <= iks_end; insert_keys_sz *= 2) 
        {
            // std::cout << insert_keys_sz << ","; std::cout << std::endl;
            size_t n = qsketch::Sketch_GPU_SF_Host_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function>::number_of_buckets
                (insert_keys_sz, sub_warp_m, work_load_factor);
            size_t search_keys_sz =  insert_keys_sz / 2;
            double max_insert_perf = 0.0;
            double max_search_perf = 0.0;
            for (size_t dimx = dimx_start; dimx <= dimx_end; dimx *= 2) {
                dim3 gridDim(insert_keys_sz / dimx, 1, 1);
                // std::cout << gridDim.x << ",";
                qsketch::Sketch_GPU_SF_Host_Sub_Warp<Key_T, Count_T, Hashed_T, Seed_T, Hash_Function> map(n,
                    sub_warp_m, times, seed_sz, qsketch::default_values::HASH_MASK_TABLE_SZ);
                map.gridDim = gridDim;
                dim3 &blockDim = map.blockDim;
                map.nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
                map.nwarps = qsketch::ceil<size_t>(map.nthreads, qsketch::default_values::WARP_SIZE);
                qsketch::Unordered_Map_GPU<Key_T, Count_T> emap;
                qsketch::Test<Key_T, Count_T> test(qsketch::curand_gpu_tool<Key_T>, qsketch::curand_gpu_tool<Count_T>, (map), (emap));
                
                run_test(max_insert_perf, max_search_perf, insert_keys_sz, search_keys_sz, map, test);
            }
            if (insert_search_test) {
                std::cout << max_insert_perf << "," << max_search_perf << std::endl;
            }
        }    
    }

    std::cout << "----------------------------------------" << std::endl;
    
    return 0;
}


int main() {

    // size_t fifo_sz;
    // cudaDeviceGetLimit(&fifo_sz,cudaLimitPrintfFifoSize);
    // // std::cout << fifo_sz << std::endl;
    // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, fifo_sz * 32);
    
    for (size_t i = 1; i <= 3; ++i) {
        std::cout << "run: " << i << std::endl;
        m = i;
        work_load_factor = 0.5;
        main1();
        work_load_factor = 1.0;
        main1();
    }
}
    