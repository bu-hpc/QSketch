#pragma once

// #include <thrust/device_ptr.h>
// #include <thrust/transform.h>
// #include <thrust/functional.h>
// #include <thrust/reduce.h>

template <typename C>
C *diff(C *c1, C *c2, size_t sz) {
    // std::cout << "sz: " << sz << std::endl;

    // thrust::device_ptr<C> dc1(to_device(c1, sz));
    // thrust::device_ptr<C> dc2(to_device(c2, sz));


    // std::cout << "d0: " << c1 << std::endl;
    // c1 = to_device(c1, sz);
    // std::cout << "d1: " << c1 << std::endl;
    // c2 = to_device(c2, sz);
    // std::cout << "d2" << std::endl;

    C *c3 = new C[sz];
    // checkKernelErrors(cudaMalloc((void **)&c3, sizeof(C) * sz));
    // thrust::device_ptr<C> dc3(c3);
    // C *c3 = new C[sz];
    // thrust::transform(dc1, dc1 + sz, dc2, dc3, thrust::minus<C>());// c3 = c1 - c2
    thrust::transform(c1, c1 + sz, c2, c3, thrust::minus<C>());// c3 = c1 - c2

    return c3;
}

template <typename C>
struct Percentage_Functor {
    __host__ __device__
    double operator()(const C &c1, const C &c2) {
        return (double(c1) / c2) * 100.0;
    }
};

template <typename C>
double *percentage(C *c1, C *c2, size_t sz) {
    // c1 = to_device(c1, sz);
    // c2 = to_device(c2, sz);

    double *p = new double[sz];
    // checkKernelErrors(cudaMalloc((void **)&p, sizeof(double) * sz));
    thrust::transform(c1, c1 + sz, c2, p, Percentage_Functor<C>()); 
    return p;
}

template <typename C>
C max_diff(C *diff, size_t sz) {
    C max = thrust::reduce(diff, diff + sz, C(0), thrust::maximum<C>());
    return max;
}

template <typename C>
double average_diff(C *diff, size_t sz) {
    double sum = thrust::reduce(diff, diff + sz, 0.0, thrust::plus<C>());
    return sum / sz;
}
