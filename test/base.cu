#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <cassert>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>


struct Base
{
    int a;
};

struct Derived : Base
{
    void print() {
        std::cout << a << std::endl;
    }
};


template <typename T>
struct BB
{
    T a = 1;
};

template <typename T>
struct B : BB<T>
{
    T a = 2;
};

// int a = 1;

template <typename T>
struct D : B<T>
{
    // using B<T>::a;
    using BB<T>::a;
    void print() {
        std::cout << a << std::endl;
    }
};

template <typename T>
struct Test
{
    T a = 100;
    Test(T _a = a) : a(_a) {}
    void print() {
        std::cout << a << std::endl;
    }
};


int main(int argc, char const *argv[])
{
    // Derived d;
    // d.a = 100;
    // d.print();
    // D<int> d;
    // D d;
    // d.a = 100;
    // d.print();
    Test<int> t;
    t.print();
    return 0;
}