#pragma once

namespace qsketch {

template <typename T>
class Random {
public:
    virtual T operator() {

    }

    virtual void operator(T *keys, size_t sz) {

    }

    virtual void set_seed() {

    }
}

}