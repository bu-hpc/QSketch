#include <iostream>

using uchar = unsigned char;
using uint = unsigned int;
using ulong = unsigned long;
using ullong = unsigned long long;

template <typename T>
constexpr size_t bits = sizeof(T) * 8; // BITS_PER_BYTE = 8;

bool set(uchar *hash_mask, uint id) {
    uint cid = id / bits<uchar>;
    uint bid = id % bits<uchar>;

    std::cout << cid << "," << bid << std::endl;

    return false;
}


int main(int argc, char const *argv[])
{
    /* code */
    set(nullptr, 2);
    std::cout << bits<uint> << std::endl;
    return 0; 
}