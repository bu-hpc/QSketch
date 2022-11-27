#pragma once

namespace qsketch {
template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T, 
    typename Hash_Function> //
struct Count_Min_Sketch_GPU_Host_Sim : Sketch<Key_T, Count_T> {

    using Insert_Kernel_Function = void (Key_T *, size_t,
        Count_T *, size_t , size_t,
        const Hash_Function &,
        Seed_T *, size_t,
        size_t,
        void *);

    using Search_Kernel_Function = void (Key_T *, size_t,
        Count_T *, size_t , size_t,
        const Hash_Function &,
        Seed_T *, size_t,
        Count_T *, Count_T,
        size_t,
        void *);

    dim3 gridDim;
    dim3 blockDim;

    size_t nwarps;
    size_t nthreads;

    // Hash_Function *hash = nullptr;
    Insert_Kernel_Function *insert_kernel = nullptr;
    Search_Kernel_Function *search_kernel = nullptr;

    size_t n;
    size_t m;
    size_t table_total_sz;
    Count_T *table = nullptr;

    size_t seed_sz;
    size_t seed_num;
    size_t seed_total_sz;

    Seed_T *seed = nullptr;

    Count_Min_Sketch_GPU_Host_Sim(
        Insert_Kernel_Function *_insert_kernel,
        Search_Kernel_Function *_search_kernel,
        size_t _n = default_hash_n, size_t _m = default_hash_m, size_t ss = default_seed_sz) : 
        n(_n), m(_m),
        seed_sz(ss) {

        insert_kernel = _insert_kernel;
        search_kernel = _search_kernel;

        seed_num = m * default_values::WARP_SIZE + 1;
        // std::cout << "seed_num: " << seed_num << std::endl;

        table_total_sz = n * m * default_values::WARP_SIZE;
        // std::cout << "table_total_sz: " << table_total_sz << std::endl;
        table = cpu_tool<Count_T>.zero(table, table_total_sz);

        seed_total_sz = seed_sz * seed_num;
        // std::cout << "seed_total_sz: " << seed_total_sz << std::endl;

        // random seeds
        seed = cpu_tool<Seed_T>.random(seed, seed_total_sz, 1, std::numeric_limits<Seed_T>::max());
        gridDim = default_values::grid_dim;
        blockDim = default_values::block_dim;

        nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
        nwarps = ceil<size_t>(nthreads, default_values::WARP_SIZE);

    }

    virtual int insert(Key_T *keys, size_t keys_sz) {
        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        insert_kernel(
                keys, keys_sz,
                table, n, m,
                Hash_Function(),
                seed, seed_sz,
                work_load_per_warp,
                nullptr
            );
        return 0;
    }
    virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
        size_t work_load_per_warp = ceil<size_t>(keys_sz, nwarps);
        search_kernel(
                keys, keys_sz,
                table, n, m,
                Hash_Function(),
                seed, seed_sz,
                count, std::numeric_limits<Count_T>::max(),
                work_load_per_warp,
                nullptr
            );
        return 0;
    }

    virtual void clear() {
        table = cpu_tool<Count_T>.zero(table, table_total_sz);
    }

    virtual ~Count_Min_Sketch_GPU_Host_Sim() {
        delete [] table;
        delete [] seed;
    }
};


}