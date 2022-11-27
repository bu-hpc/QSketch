#include <iostream>

using namespace std;

int main(int argc, char const *argv[])
{
    unsigned int i;
    unsigned int cur = 0;
    while (cin >> i) {
        if (i != (cur)) {
            std::cout << "err: " << i << ", " << cur << endl;
        }

        cur ++;
    }
    std::cout << "ok" << std::endl;
    return 0;
}