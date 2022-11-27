#include <iostream>
#include <vector>

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

int main(int argc, char const *argv[])
{
    if (argc != 2) {
        return -1;
    }

    size_t sz = std::stoul(argv[1]);

    auto v = cal_primes<ulong>(sz);

    std::cout << v.size() << std::endl;
    std::cout << v.back() << std::endl;

    return 0;
}