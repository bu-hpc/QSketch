#include <iostream>
#include <typeinfo>

using namespace std;

template <typename T>
T fun(T t) {
    cout << "fun: " << typeid(T).name() << endl;
    return t;
}

template <typename T>
T gun(T t) {
    cout << "gun: " << typeid(T).name() << endl;
    return t;
}


template <typename T>
void test(T (*ptr)(T), T t) {
    ptr(t);
}

struct MyClass
{
    
};


int main(int argc, char const *argv[])
{
    int a = 0;
    double d = 0;
    MyClass my;
    test(fun<int>, a);
    test(gun<double>, d);

    test(fun<MyClass>, my);
    test(gun<MyClass>, my);
    return 0;
}