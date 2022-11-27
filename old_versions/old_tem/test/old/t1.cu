#include <iostream>
#include "h1.h"
#include "h2.h"

using namespace std;

int main(int argc, char const *argv[])
{

    test<int>.val = 100;
    print<int>();
    set<int>(101);
    print<int>();

    test2();

    print<int>();
    return 0;
}