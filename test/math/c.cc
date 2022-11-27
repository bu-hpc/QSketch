#include <iostream>

using namespace std;

long A(long m, long n) {
    long ans = m;
    for (long i = 1; i < n; ++i) {
        ans *= m - i;
    }
    return ans;
}

long C(long m, long n) {
    return A(m, n) / A(m - n, m - n);
}

int main(int argc, char const *argv[])
{
    // cout << C(4, 2) << endl;
    for (long i = 0; i <= 32; ++i) {
        cout << i << ":" << C(32, i) << endl;
    }
    return 0;
}