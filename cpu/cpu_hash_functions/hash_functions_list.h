#ifndef CPU_HASH_FUNCTIONS_LIST
#define CPU_HASH_FUNCTIONS_LIST
#include "lib.h"

// template <typename KEY_TYPE, typename HASH_TYPE, typename SEED_TYPE = default_hash_function_seed_type>
// struct Hash {
//     SEED_TYPE seed;

//     Hash() = default;

//     Hash(const SEED_TYPE &_s): seed(_s) {

//     }

//     virtual bool set_seed(const SEED_TYPE &_s) {
//         seed = _s;   
//     }

//     virtual HASH_TYPE operator()(const KEY_TYPE &) = 0;

// };

// template <typename T, typename SEED_TYPE>
// struct Hash_Mul : public Hash<T, T, SEED_TYPE> {
//     virtual T operator()(const T &key) {
//         T hash = key;
//         for (auto &v : seed) {
//             hash = hash * v;
//         }
//         return hash;
//     }

// };


// template <typename T>
// struct Hash_Mod : public Hash<T, T, T> {
//     virtual T operator()(const T &key) {
//         return key % Hash::seed;
//     }

// };

// template <typename T, typename SEED_TYPE>
// struct Hash_Mul_Mod : public Hash<T, T, SEED_TYPE> {
//     virtual T operator()(const T &key) {
//         T hash = key;
//         for (auto &v : Hash::seed.second) {
//             hash = (hash * v) % seed.first;
//         }
//         return hash;
//     }

// };


template <typename KEY_TYPE, typename HASH_TYPE, typename SEED_TYPE = default_hash_function_seed_type>
struct Hash
{
    using Functor_Type = std::function<HASH_TYPE(const KEY_TYPE &, SEED_TYPE &)>;
    Functor_Type functor;

    SEED_TYPE seed;

    Hash(const SEED_TYPE &_s, const Functor_Type &_f) : seed(_s), functor(_f) {}

    bool set_functor(const Functor_Type &_f) {
        functor = _f;
        return true;
    }

    bool set_seed(const SEED_TYPE &_s) {
        seed = _s;
        return true;
    }

    HASH_TYPE operator()(const KEY_TYPE &key) {
        return functor(key, seed);
    }
};

template <typename T, typename SEED_TYPE>
T hash_mul(const T &key, SEED_TYPE &seed) {
    T hash = key;
    for (auto &v : seed) {
        hash = hash * v;
    }
    return hash;
}

#endif
