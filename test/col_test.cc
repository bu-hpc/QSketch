#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

using namespace std;

#include "lib.h"
#include "cpu/cpu_hash_functions/hash_functions_list.h"

using T = unsigned int;
using C = unsigned int;

size_t total = 0;
T m = 65536;

template <typename H>
T col_test(H &h, T sz = std::numeric_limits<T>::max()) {
    vector<C> count(sz, 0);

    T tm = std::numeric_limits<T>::max();
    T i = tm;
    T per = tm / 10 + 1;
    do {
        if (i % per == 0) {
            cout << (i / per) << " % " << endl;
        }
        ++count[h(i) % m];
        --i;
    } while (i != 0);

    for (auto &v : count) {
        total += v;
    }
    cout << "total:\t" << total << endl;
    return *max_element(count.begin(), count.end());
}

int main(int argc, char const *argv[])
{
    std::vector<T> v;

    std::default_random_engine gen;
    std::uniform_int_distribution<T> dis(1, std::numeric_limits<T>::max());

    for (int i = 0; i < 2; ++i)
    {
        v.push_back(dis(gen));
        cout << v.back() << ",";
    }
    cout << endl;

    Hash<T, T, vector<T>> hm(v, hash_mul<T, vector<T>>);

    cout << "max: " << col_test(hm, m) << endl;


    return 0;
}