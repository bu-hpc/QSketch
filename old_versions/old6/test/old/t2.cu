#include "h1.h"
#include "h2.h"
#include <iostream>

using namespace std;

void test2() {
    test<int>.val = 1000;
    print<int>();
    set<int>(1001);
    print<int>();
}