#include <iostream>
#include <random>
#include <unordered_map>
#include <limits>

using namespace std;

int main(int argc, char const *argv[])
{
    size_t count = 512 * 1024 * 1024;
    std::default_random_engine eng;
    std::uniform_int_distribution<unsigned int> dis(0, std::numeric_limits<unsigned int>::max());
    unordered_map<unsigned int, unsigned int> um;
    for (size_t i = 0; i < count; ++i)
    {
        if (i % (1024 * 1024) == 0) {
            cout << double(i)/count * 100 << " %" << endl;
        }

        um[dis(eng)]++;
        /* code */
    }
    return 0;
}