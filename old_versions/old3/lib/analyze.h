#pragma once

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

template <typename C>
struct Percentage_Functor {
    __host__ __device__
    float operator()(const C &c1, const C &c2) {
        return (float(c1) / c2) * 100.0f;
    }
};

template <typename C>
thrust::device_vector<float> percentage(const thrust::device_vector<C> &dc1, const thrust::device_vector<C> &dc2) {
    thrust::device_vector<float> p;

    // thrust::device_vector<float> p(dc1.size());
    // double *p = new double[sz];
    // checkKernelErrors(cudaMalloc((void **)&p, sizeof(double) * sz));
    // thrust::transform(dc1.begin(), dc1.end(), dc2.begin(), p.begin(), Percentage_Functor<C>()); 
    return p;
}

template <typename C>
C max_diff(thrust::device_vector<C> &diff) {
    C max;
    // C max = thrust::reduce(diff.begin(), diff.end(), C(0), thrust::maximum<C>());
    return max;
}

template <typename C>
float average_diff(thrust::device_vector<C> &diff) {
    double sum;
    // float sum = thrust::reduce(diff.begin(), diff.end(), 0.0f, thrust::plus<C>());
    return sum / diff.size();
}
