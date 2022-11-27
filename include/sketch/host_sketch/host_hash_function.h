#pragma once

namespace qsketch {

template <typename Seed_T>
struct Hash {
    Seed_T *seed;
    size_t seed_sz;

    Hash() = default;

    Hash(Seed_T *s, size_t sz) : seed(s), seed_sz(sz) {

    }

    void set_seed(Seed_T *s, size_t sz) {
        seed = s;
        seed_sz = sz;
    }

    template <typename Key_T, typename Hashed_T = size_t>
    Hashed_T hash_mul_add(const Key_T &key) {
        Hashed_T hv = 0;
        for (size_t i = 0; i < seed_sz; ++i)
        {
            hv += key * seed[i];
        }
        return hv;
    }
};


}