#pragma once

template <typename T>
struct GPU_Tools : Tools<T> {

    T *random(T *keys, size_t keys_sz, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max()) {

        if (keys == nullptr) {
            keys = new T[keys_sz];
            // checkKernelErrors(cudaMallocHost(&keys, sizeof(T) * keys_sz));
        }

        std::uniform_int_distribution<T> dis(min, max);

        for (size_t i = 0; i < keys_sz; ++i)
        {
            keys[i] = dis(eng);
        }

        return keys;
    }

    T *zero(T *keys, size_t keys_sz) {
        if (keys == nullptr) {
            // keys = new T[keys_sz];
            checkKernelErrors(cudaMalloc(&keys, sizeof(T) * keys_sz));
        }
        cudaMemset(keys, 0, sizeof(T) * keys_sz);
        return keys;
    }
};

template <typename Key_T, typename Count_T, typename Hashed_T = size_t> // 
struct Count_Min_Sketch_GPU : Sketch<Key_T, Count_T> {

    dim3 gridDim;
    dim3 blockDim;

    Count_T **hash_tables;
    size_t hash_tables

    virtual int insert(Key_T *keys, size_t keys_sz) {
        for (size_t i = 0; i < keys_sz; ++i) {
            ++um[keys[i]];
        }
        return 0;
    }
    virtual int search(Key_T *keys, size_t keys_sz, Count_T *count) {
        for (size_t i = 0; i < keys_sz; ++i) {
            count[i] = um[keys[i]];
        }
        return 0;
    }
};
