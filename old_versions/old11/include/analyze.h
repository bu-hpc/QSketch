#pragma once

using Percentage_T = double;

template <typename C>
thrust::device_vector<C> diff(const thrust::device_vector<C> &dc1, const thrust::device_vector<C> &dc2) {

    thrust::device_vector<C> dc3(dc1.size());
    thrust::transform(dc1.begin(), dc1.end(), dc2.begin(), dc3.begin(), thrust::minus<C>());

    return dc3;
}

template <typename C>
struct Percentage_Functor {
    __host__ __device__
    Percentage_T operator()(const C &c1, const C &c2) {
        return (Percentage_T(c1) / c2) * 100.0f;
    }
};

template <typename C>
thrust::device_vector<Percentage_T> percentage(const thrust::device_vector<C> &dc1, const thrust::device_vector<C> &dc2) {

    thrust::device_vector<Percentage_T> p(dc1.size());
    thrust::transform(dc1.begin(), dc1.end(), dc2.begin(), p.begin(), Percentage_Functor<C>()); 
    return p;
}

template <typename C>
C max_diff(thrust::device_vector<C> &diff) {
    C max = thrust::reduce(diff.begin(), diff.end(), C(0), thrust::maximum<C>());
    return max;
}

template <typename C>
Percentage_T average_diff(thrust::device_vector<C> &diff) {
    float sum = thrust::reduce(diff.begin(), diff.end(), 0.0f, thrust::plus<C>());
    return sum / diff.size();
}

template <typename C>
C sum(thrust::device_vector<C> &diff) {
    C max = thrust::reduce(diff.begin(), diff.end(), C(0), thrust::plus<C>());
    return max;
}


template <typename C>
struct Count_Greater
{
    __host__ __device__
    bool operator()(const C &c) {
        // return c > 4294967000u;
        return c > 4000000000u;
    }
};

template <typename C>
C greater(thrust::device_vector<C> &diff) {
    C count = thrust::count_if(diff.begin(), diff.end(), Count_Greater<C>());
    // thrust::count_if(thrust::device, vec.begin(), vec.end(), is_odd());
    return count;
}
