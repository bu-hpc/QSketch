#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <bitset>

#define WARP_SIZE 32

using namespace std;

static unsigned int warp[WARP_SIZE];
std::default_random_engine eng;
std::uniform_int_distribution<unsigned int> dis(1, std::numeric_limits<unsigned int>::max());

unsigned int insert() {
    unsigned int mask = dis(eng) & dis(eng) & dis(eng) | (1u << (dis(eng) & 31));
    // cout << bitset<32>(mask) << endl;
    unsigned int m = 1;
    unsigned int min = std::numeric_limits<unsigned int>::max();
    for (size_t i = 0; i < WARP_SIZE; ++i) {
        if (m & mask) {
            min = std::min(min, warp[i]);
        }
        m = m << 1;
    }

    unsigned int new_min = min + 1;
    // cout << new_min << endl;
    m = 1;
    for (size_t i = 0; i < WARP_SIZE; ++i) {
        if (m & mask) {
            if (warp[i] <= new_min) {
                warp[i] = new_min;
            }
        }
        m = m << 1;
    }

    // for (int i = 0; i < WARP_SIZE; ++i)
    // {
    //     cout << warp[i] << " ";
    // }

    return min;
}

void warp_test(size_t insert_keys_size) {
    for (size_t i = 0; i < insert_keys_size; ++i) {
        cout << i << ": " << insert() << endl;
    }
}

int main(int argc, char const *argv[])
{
    warp_test(100);
    return 0;
}