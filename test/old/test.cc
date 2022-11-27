#include <thrust/device_vector.h>
#include <iostream>

template <typename C>
thrust::device_vector<C> diff(const thrust::device_vector<C> &dc1, const thrust::device_vector<C> &dc2) {
    // dc1.size();
    thrust::device_vector<int> dc3(100);
    // thrust::device_vector<C> dc3(dc1.size());
    // thrust::transform(dc1.begin(), dc1.end(), dc2.begin(), dc3.begin(), thrust::minus<C>());

    thrust::device_vector<C> ans;
    return ans;
    // return dc3;
}

template <typename Count_T>
int search_test() {
    Count_T *counts = nullptr, *e_counts = nullptr;
    size_t search_keys_sz = 100;

    auto dc3 = diff(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
                thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
    return 0;
}

 int main( int argc, char const *argv[])
{
    search_test<unsigned int>();
    // unsigned int *p;
    // size_t sz = 100;
    // thrust::device_vector<unsigned int> dv1, dv2;
    // auto dc3 = diff(thrust::device_vector<unsigned int>(p, p + sz), 
    //             thrust::device_vector<unsigned int>(p, p + sz));
    // thrust::device_vector<unsigned int> tdv(100);

    // unsigned int *hp = new unsigned int[100];
    // for (unsigned int i = 0; i < 100; ++i) {
    //     hp[i] = i + 1;
    // }
    // thrust::device_vector<unsigned int> dv1(hp, hp + 100);
    // std::cout << thrust::reduce(dv1.begin(), dv1.end(), 0, thrust::plus<unsigned int>()) << std::endl;

    // unsigned int *dp;
    // cudaMalloc((void **)&dp, sizeof(unsigned int) * 100);
    // cudaMemcpy(dp, hp, sizeof(unsigned int) * 100, cudaMemcpyHostToDevice);
    // thrust::device_vector<unsigned int> dv2(dp, dp + 100);
    // std::cout << thrust::reduce(dv2.begin(), dv2.end(), 0, thrust::plus<unsigned int>()) << std::endl;
    return 0;
}