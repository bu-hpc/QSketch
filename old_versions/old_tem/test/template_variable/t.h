
template<typename T>
struct Data{
    T val;
    void test() {

    }
};

// template<>
// struct Data<unsigned int> {
//     unsigned int val;
//     unsigned int a;
// };


template<>
void Data<unsigned int>::test() {
    int a;
}

template<typename T>
Data<T> tv;