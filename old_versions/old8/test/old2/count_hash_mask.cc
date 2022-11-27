#include <iostream>
#include <unordered_map>
#include <bitset>

using namespace std;

int main(int argc, char const *argv[])
{
    string str;
    unsigned int mask;
    unordered_map<unsigned int, size_t> um;
    while (cin >> str >> mask) {
        um[mask]++;
    }
    for (auto &v : um) {
        bitset<32> bs(v.first);
        std::cout << v.first << " : " << v.second << "\t<--" << bs.count() << " --> \t" << bs << std::endl;
    }

    return 0;
}