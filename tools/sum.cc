#include <iostream>
using namespace std;

int main(int argc, char const *argv[])
{
    long sum = 0;
    int a;
    while (cin >> a) {
        sum += a;
    }
    cout << sum << endl;
    return 0;
}