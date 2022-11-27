#pragma once

namespace qsketch {

template <typename Key_T, typename Count_T> // 
struct Sketch {
    virtual int insert(Key_T *, size_t) = 0;
    virtual int search(Key_T *, size_t, Count_T *) = 0;

    virtual unsigned char *pre_cal(Key_T *, size_t, void *) {
        return nullptr;
    }

    virtual void clear() = 0;


    virtual ~Sketch() {
        
    }
#ifdef QSKETCH_DEBUG
    virtual void print(std::ostream &) {

    }
    virtual void show_memory_usage(std::ostream &) {

    }
#endif
};

}
// #include "default.h"
#include "host_sketch/host_sketch.h"
#include "device_sketch/device_sketch.h"

namespace qsketch {

template <typename Seed_T>
int copy_seed_to_device(Host_Seed<Seed_T> &hs, Device_Seed<Seed_T> &ds) {
    ds.free();
    ds.seed_num = hs.seed_num;
    ds.seed_sz = hs.seed_sz;
    ds.seed_total_sz = hs.seed_sz;
    gpu_tool<Seed_T>.zero(ds.seed, ds.seed_total_sz);
    CUDA_CALL(cudaMemcpy(ds.seed, hs.seed, sizeof(Seed_T) * hs.seed_total_sz, cudaMemcpyHostToDevice));
    return 0;
}

template <typename Seed_T>
int copy_seed_to_host(Host_Seed<Seed_T> &hs, Device_Seed<Seed_T> &ds) {
    hs.free();
    hs.seed_num = ds.seed_num;
    hs.seed_sz = ds.seed_sz;
    hs.seed_total_sz = ds.seed_sz;
    cpu_tool<Seed_T>.zero(hs.seed, hs.seed_total_sz);
    CUDA_CALL(cudaMemcpy(hs.seed, ds.seed, sizeof(Seed_T) * hs.seed_total_sz, cudaMemcpyDeviceToHost));
    return 0;
}

}
