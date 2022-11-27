#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>


using T = unsigned int;
using C = unsigned int;

using ull = unsigned long long;


int main(int argc, char const *argv[])
{
	T sc = 43850;
	T sz = 65536;

	std::vector<T> count(sz, 0);
	std::unordered_map<T, T> um;

	for (T i = 0; i < sz; ++i) {
		T val = sc * i % sz;
		count[val]++;
		std::cout << i << ":" << val << std::endl;
	}

	for (auto &v : count) {
		um[v]++;
		// std::cout << v << std::endl;
	}

	for (auto &u : um) {
        std::cout << u.first << "\t:\t" << u.second << std::endl;
    }

	return 0;
}