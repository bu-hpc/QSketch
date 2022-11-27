#include "qsketch.h"

int main(int argc, char const *argv[])
{
    qsketch::test_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    qsketch::Test_Struct t;
    return 0;
}