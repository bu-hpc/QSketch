#include "lib.cuh"

// test::Test_kernel1<int> a;

// __global__ void kernel_1() {
//     size_t tid = threadIdx.x;
//     if (tid == 0) {
//         printf("kernel_1\n");
//     }
// }


int main(int argc, char const *argv[])
{
    test::Test<int> t;
    t.run();

    test::Test_kernel1<int> t1;
    t1.run();

    test::Test<int> *ptr = &t1;
    test::Test<int> &r = t1;

    ptr->run();
    r.run();

    test::Test<int> *ptr2 = new test::Test_kernel2<int>;
    ptr2->run();
    return 0;
}