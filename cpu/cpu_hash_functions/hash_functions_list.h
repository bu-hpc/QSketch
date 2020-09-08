#ifndef CPU_HASH_FUNCTIONS_LIST
#define CPU_HASH_FUNCTIONS_LIST
#include "../lib.h"

template <typename KEY_TYPE, typename HASH_TYPE, typename SEED_TYPE = default_hash_function_seed_type>
struct Hash;

template <typename KEY_TYPE, typename HASH_TYPE, typename SEED_TYPE = default_hash_function_seed_type>
struct Hash {
    SEED_TYPE seed;

    Hash() = default;

    Hash(const SEED_TYPE &_s): seed(_s) {

    }

    virtual bool set_seed(const SEED_TYPE &_s) {
        seed = _s;   
    }

    virtual HASH_TYPE operator()(const KEY_TYPE &) = 0;

};

template <typename T>
struct Hash_Mod : public Hash<T, T, T> {

    virtual T operator()(const T &key) {
        return key % seed;
    }

};

template <typename T>
struct Hash_Mul_Mod : public Hash<T, T, T> {

    virtual T operator()(const T &key) {
        T hash = 0;
        for (auto &v : seed.second) {
            hash = (hash * v) % seed.first;
        }
        return hash;
    }

};

template <typename KEY_TYPE, typename HASH_TYPE, typename SEED_TYPE = default_hash_function_seed_type>
struct Hash_Functor : public Hash<KEY_TYPE, HASH_TYPE, SEED_TYPE>
{
    using Functor_Type = std::function<HASH_TYPE(const KEY_TYPE &, SEED_TYPE &)>;
    Functor_Type functor;

    Hash_Functor(const SEED_TYPE &_s, const Functor_Type &_f) : Hash(_s), functor(_f) {}

    set_functor(const Functor_Type &_f) {
        functor = _f;
    }

    virtual T operator()(const KEY_TYPE &key) {
        return functor(key, seed);
    }
};


#endif
