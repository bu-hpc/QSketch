#pragma once

#include <iostream>

#define HELLO_WORLD "hello world!"


template <typename T>
void test () {
    T val;
    std::cout << HELLO_WORLD << std::endl;

}


void header_fun();