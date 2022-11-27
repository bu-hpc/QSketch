#include <iostream>
#include <vector>
#include <string>
#include <random>


int main(int argc, char const *argv[])
{
    // if (argc != 2) {
    //     std::cout << "usage: ./" << argv[0] << " n" << std::endl;
    //     // std::cout << "calcualte the greatest prime number which is not greater than n" << std::endl;
    //     return -1;
    // }

    size_t sz = std::stoul(argv[1]);
    size_t count = std::stoul(argv[2]);

    std::vector<size_t> v(sz, 0);
    std::default_random_engine gen;
    std::uniform_int_distribution<size_t> dis(0, sz - 1);
    for (size_t i = 0; i < count; ++i) {
        v[dis(gen)]++;
    }
    // size_t c = 0;
    // std:;unordered_map<size_t, size_t> um;
    std::vector<size_t> m(256, 0);
    for (auto &val : v) {
        // if (val >= 1) {
        //     c++;
        // }
        m[val]++;
    }
    for (size_t i = 0; i < m.size(); ++i) {
        std::cout << i << ":" << m[i] << std::endl;
    }
    // std::cout << c << std::endl;
    return 0;
}