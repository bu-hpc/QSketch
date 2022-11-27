#include <iostream>
#include <vector>
#include <string>

template <typename T>
std::vector<T> cal_primes(size_t sz) {
    std::vector<bool> buf(sz + 1, true);
    std::vector<T> ps;

    buf[0] = buf[1] = false;

    for (size_t i = 2; i <= sz; ++i) {
        if (buf[i]) {
            ps.push_back(i);
            for (size_t j = i + i; j <= sz; j += i) {
                buf[j] = false;
            }
        }
        
    }

    return ps;

}

using ulong = unsigned long;

using std::cout;
using std::endl;

int main(int argc, char const *argv[])
{
    {
        size_t iks_start = 32; // iks  : insert_keys_sz
        size_t iks_end = 256 * 1024 * 1024;
        auto v = cal_primes<ulong>(3 * 128 * 1024 * 1024);
        // for (size_t sz = iks_start; sz <= iks_end; sz *= 2) {
        //     auto v = cal_primes<ulong>(sz);
        //     std::cout
        //          // << sz << ", " 
        //         << v.back() << "," << std::endl;
        // }

        size_t a = 256 * 1024 * 1024;
        size_t b = 3 * 128 * 1024 * 1024;

        for (auto it = v.rbegin(); it != v.rend(); ++it) {
            if (*it < a) {
                std::cout << (*it) << "," << std::endl;
                a = a / 2;
            }
            if (*it < b) {
                std::cout << (*it) << "," << std::endl; 
                b = b / 2;
            }
        }

        // std::cout << v.size() << std::endl;
    }
    return 0;

    if (argc != 2) {
        std::cout << "usage: ./" << argv[0] << " n" << std::endl;
        std::cout << "calcualte the greatest prime number which is not greater than n" << std::endl;
        return -1;
    }

    size_t sz = std::stoul(argv[1]);

    auto v = cal_primes<ulong>(sz);

    // std::cout << v.size() << std::endl;
    std::cout << v.back() << std::endl;

    return 0;
}