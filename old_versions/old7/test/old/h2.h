#pragma once
#include <iostream>

template <typename T>
void print() {
    std::cout << test<T>.val << std::endl;
}

template <typename T>
void set(T t) {
    test<T>.val = t;
}


void test2();