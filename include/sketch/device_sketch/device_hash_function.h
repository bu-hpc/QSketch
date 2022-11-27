#pragma once

namespace qsketch {
template <typename Key_T, typename Hashed_T = size_t, typename Seed_T = size_t>
struct hash_mul_add {
    __device__ __host__ Hashed_T operator()(Seed_T *seed, size_t s_sz, const Key_T &key) const{
        Hashed_T hv = 0;
        // printf("hash_mul_add\n");
        #pragma unroll
        for (size_t i = 0; i < s_sz; ++i)
        {
            // printf("si: %lu\n", i);
            // std::cout << i << std::endl;
            hv += key * seed[i];
        }
        // printf("hv: %u\n", hv);
        return hv;
    }

    Hashed_T host_hash(Seed_T *seed, size_t s_sz, const Key_T &key) const{
        Hashed_T hv = 0;
        // printf("hash_mul_add\n");
        #pragma unroll
        for (size_t i = 0; i < s_sz; ++i)
        {
            // printf("si: %lu\n", i);
            // std::cout << i << std::endl;
            hv += key * seed[i];
        }
        // printf("hv: %u\n", hv);
        return hv;
    }
};

template <typename Key_T, typename Hashed_T = size_t, typename Seed_T = size_t>
struct hash_xor {
    __device__ __host__ Hashed_T operator()(Seed_T *seed, size_t s_sz, const Key_T &key) const{
        Hashed_T hv = key;
        #pragma unroll
        for (size_t i = 0; i < s_sz; ++i)
        {
            // hv += key * seed[i];
            hv = hv ^ seed[i];
        }
        return hv;
    }
};



template <typename Key_T, typename Hashed_T = size_t, typename Seed_T = size_t>
struct hash_fast {
    __device__ __host__ Hashed_T operator()(Seed_T *seed, size_t s_sz, const Key_T &key) const{
        return key;
    }
};

}